// image-namer-standalone.js - Fully standalone image renamer
// Just run: node image-namer-standalone.js <folder_path>
// Model downloads automatically on first run (~100MB)

const fs = require('fs').promises;
const path = require('path');

// ===== CONFIGURATION =====
// Get folder from command line argument (required)
const IMAGES_FOLDER = process.argv[2];

// Dynamic import for transformers (ES module)
let pipeline, RawImage;

// ================= GPU CHANGE =================
const ort = require('onnxruntime-node');

function detectDevice() {
  try {
    const providers = ort.getAvailableExecutionProviders();
    if (providers.includes('CUDAExecutionProvider')) {
      console.log('🚀 CUDA detected — using GPU acceleration');
      return 'cuda';
    }
  } catch {
    // ignore
  }
  console.log('💻 CUDA not available — using CPU');
  return 'cpu';
}
// ==============================================

async function loadTransformers() {
  console.log('📦 Loading AI libraries...');

  // Suppress ONNX runtime warnings
  process.env.ORT_LOG_SEVERITY_LEVEL = '5';  // Only show errors

  const transformers = await import('@xenova/transformers');
  pipeline = transformers.pipeline;
  RawImage = transformers.RawImage;
  console.log('✅ Libraries loaded!\n');
}

// ===== HELPER FUNCTIONS =====

// Recursively get all files in directory and subdirectories
async function getAllFiles(dirPath, fileList = []) {
  try {
    const entries = await fs.readdir(dirPath, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dirPath, entry.name);

      if (entry.isDirectory()) {
        // Recursively process subdirectories
        await getAllFiles(fullPath, fileList);
      } else if (entry.isFile()) {
        fileList.push(fullPath);
      }
    }
  } catch (error) {
    console.error(`  ⚠️  Cannot access directory: ${dirPath}`);
  }

  return fileList;
}

// Check if a file is an image based on extension
function isImageFile(filename) {
  const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'];
  const ext = path.extname(filename).toLowerCase();
  return imageExtensions.includes(ext);
}

// Check if current filename is already good enough
function isFilenameGoodEnough(currentFilename, aiDescription) {
  const ext = path.extname(currentFilename).toLowerCase();
  const nameWithoutExt = path.basename(currentFilename, ext).toLowerCase();

  // Skip generic names like "img_1234", "dsc_5678", "screenshot", etc.
  const genericPatterns = [
    /^img[-_]?\d+$/i,           // img123, img_123, img-123
    /^dsc[-_]?\d+$/i,           // dsc123, dsc_123
    /^image[-_]?\d+$/i,         // image123
    /^photo[-_]?\d+$/i,         // photo123
    /^screenshot/i,             // screenshot*
    /^pic[-_]?\d+$/i,           // pic123
    /^\d{8}[-_]\d{6}$/,         // 20231231_123456 (phone camera format)
    /^[a-f0-9]{8,}$/i,          // long hex strings (random IDs)
    /^untitled/i,               // untitled
    /^new[-_]?image/i,          // new-image, newimage
  ];

  // If filename matches generic pattern, it needs renaming
  for (const pattern of genericPatterns) {
    if (pattern.test(nameWithoutExt)) {
      return false;
    }
  }

  // Extract meaningful words from both filename and description
  const descWords = aiDescription
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(word => word.length > 2); // Only words longer than 2 chars

  const nameWords = nameWithoutExt
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(word => word.length > 2);

  // If filename has descriptive words that overlap with AI description
  // consider it good enough (at least 2 word matches or 40% overlap)
  let matchCount = 0;
  for (const descWord of descWords) {
    for (const nameWord of nameWords) {
      // Check for substring match (e.g., "cat" matches "cats")
      if (descWord.includes(nameWord) || nameWord.includes(descWord)) {
        matchCount++;
        break;
      }
    }
  }

  // Consider good if:
  // - At least 2 words match, OR
  // - 40% or more of description words are in the filename
  const overlapRatio = descWords.length > 0 ? matchCount / descWords.length : 0;
  return matchCount >= 2 || overlapRatio >= 0.4;
}

// Create a safe filename from AI description
function createSafeFilename(description, originalFilename) {
  let safeName = description
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '')     // Remove special chars
    .replace(/\s+/g, '-')              // Spaces to hyphens
    .replace(/^-+|-+$/g, '')           // Remove leading/trailing hyphens
    .replace(/-+/g, '-')               // Multiple hyphens to single
    .substring(0, 50);                 // Limit length

  const ext = path.extname(originalFilename);
  return safeName + ext;
}

// Initialize the AI model (with progress indicator)
async function initializeModel() {
  console.log('🤖 Initializing AI model...');
  console.log('   (First run will download ~100MB model - please wait)\n');

  try {
    // ================= GPU CHANGE =================
    const device = detectDevice();
    // ==============================================

    // Use vit-gpt2 model (reliable and available)
    const captioner = await pipeline(
      'image-to-text',
      'Xenova/vit-gpt2-image-captioning',
      {
        // ================= GPU CHANGE =================
        device,
        dtype: device === 'cuda' ? 'fp16' : 'fp32'
        // ==============================================
      }
    );

    console.log('✅ Model ready!\n');
    return captioner;

  } catch (error) {
    throw new Error(`Failed to load model: ${error.message}`);
  }
}

// Get description from AI
async function getImageDescription(captioner, imagePath) {
  console.log(`  🤖 Analyzing image...`);

  try {
    // Let the AI decide - natural token length
    const result = await captioner(imagePath, {
      max_new_tokens: 15,   // Reasonable limit but not restrictive
      num_beams: 3,
      do_sample: false
    });

    // Extract and clean
    let description = result[0].generated_text;

    // Simple cleanup - just remove common prefixes and punctuation
    description = description
      .toLowerCase()
      .replace(/^(a |an |the |this |that |this is |there is )/gi, '')
      .replace(/[.,;:!?]+$/g, '')
      .trim();

    return description || 'image';

  } catch (error) {
    throw new Error(`Analysis failed: ${error.message}`);
  }
}

// ===== MAIN SCRIPT =====

async function renameImagesInFolder() {
  console.log('🖼️  Standalone Image Namer\n');

  // Check if directory argument was provided
  if (!IMAGES_FOLDER) {
    console.log('❌ Error: No directory specified!\n');
    console.log('Usage: node image-namer-standalone.js <folder_path>\n');
    console.log('Examples:');
    console.log('  node image-namer-standalone.js ./my-photos');
    console.log('  node image-namer-standalone.js /home/user/pictures');
    console.log('  node image-namer-standalone.js "C:\\Users\\Name\\Pictures"');
    console.log('  node image-namer-standalone.js .\n');
    return;
  }

  // Show help if requested
  if (IMAGES_FOLDER === '--help' || IMAGES_FOLDER === '-h') {
    console.log('Usage: node image-namer-standalone.js <folder_path>\n');
    console.log('This tool uses AI to analyze images and rename them based on their content.\n');
    console.log('Examples:');
    console.log('  node image-namer-standalone.js ./my-photos');
    console.log('  node image-namer-standalone.js /absolute/path/to/images');
    console.log('  node image-namer-standalone.js "C:\\Users\\Name\\Pictures"');
    console.log('  node image-namer-standalone.js .\n');
    return;
  }

  console.log('📝 No installation required - everything runs in Node.js!\n');

  try {
    // Load the transformers library
    await loadTransformers();

    // Initialize AI model
    const captioner = await initializeModel();

    // Check if images folder exists
    try {
      const stats = await fs.stat(IMAGES_FOLDER);
      if (!stats.isDirectory()) {
        console.log(`❌ Error: "${IMAGES_FOLDER}" is not a directory!\n`);
        return;
      }
    } catch {
      console.log(`❌ Error: Directory not found: ${path.resolve(IMAGES_FOLDER)}\n`);
      console.log('Make sure the path is correct and the directory exists.\n');
      return;
    }

    // Read all files in the directory recursively
    console.log(`📁 Scanning directory (including subdirectories): ${IMAGES_FOLDER}\n`);
    const allFiles = await getAllFiles(IMAGES_FOLDER);

    // Filter to only image files
    const imageFiles = allFiles.filter(isImageFile);
    console.log(`Found ${imageFiles.length} image(s)\n`);

    if (imageFiles.length === 0) {
      console.log('❌ No images found in the directory or its subdirectories!');
      console.log(`   Add .jpg, .png, or other images to: ${path.resolve(IMAGES_FOLDER)}\n`);
      return;
    }

    // Process each image
    let renamed = 0;
    let skipped = 0;
    let errors = 0;

    for (let i = 0; i < imageFiles.length; i++) {
      const imagePath = imageFiles[i];
      const filename = path.basename(imagePath);
      const directory = path.dirname(imagePath);

      console.log(`[${i + 1}/${imageFiles.length}] ${imagePath}`);

      try {
        // Get AI description
        const description = await getImageDescription(captioner, imagePath);
        console.log(`  💡 Description: "${description}"`);

        // Check if current filename is already good enough
        if (isFilenameGoodEnough(filename, description)) {
          console.log(`  ✓ Current name is already descriptive - skipping\n`);
          skipped++;
          continue;
        }

        // Create new filename
        const newFilename = createSafeFilename(description, filename);
        const newPath = path.join(directory, newFilename);

        // Rename if different
        if (newFilename !== filename) {
          // Check if target exists
          try {
            await fs.access(newPath);
            // File exists! Add a number to make it unique
            let counter = 1;
            let uniquePath = newPath;
            let uniqueFilename = newFilename;

            while (true) {
              const ext = path.extname(newFilename);
              const nameWithoutExt = newFilename.slice(0, -ext.length);
              uniqueFilename = `${nameWithoutExt}-${counter}${ext}`;
              uniquePath = path.join(directory, uniqueFilename);

              try {
                await fs.access(uniquePath);
                // Still exists, try next number
                counter++;
              } catch {
                // Doesn't exist - we can use this!
                break;
              }
            }

            await fs.rename(imagePath, uniquePath);
            console.log(`  ✅ Renamed to: ${uniqueFilename} (added -${counter} to avoid duplicate)\n`);
            renamed++;
          } catch {
            // Safe to rename - doesn't exist
            await fs.rename(imagePath, newPath);
            console.log(`  ✅ Renamed to: ${newFilename}\n`);
            renamed++;
          }
        } else {
          console.log(`  ℹ️  Name unchanged\n`);
          skipped++;
        }

      } catch (error) {
        console.error(`  ❌ Error: ${error.message}\n`);
        errors++;
      }
    }

    // Summary
    console.log('━'.repeat(50));
    console.log('🎉 Complete!\n');
    console.log(`   ✅ Renamed: ${renamed}`);
    console.log(`   ⏭️  Skipped: ${skipped}`);
    if (errors > 0) {
      console.log(`   ❌ Errors:  ${errors}`);
    }
    console.log('');

  } catch (error) {
    console.error('❌ Fatal error:', error.message);

    if (error.message.includes('ENOENT')) {
      console.log('\n📝 Make sure the images folder exists!');
    }
  }
}

// Check Node version
const nodeVersion = parseInt(process.version.slice(1).split('.')[0]);
if (nodeVersion < 18) {
  console.error('❌ This script requires Node.js 18 or higher');
  console.log(`   Your version: ${process.version}`);
  console.log('   Download latest from: https://nodejs.org\n');
  process.exit(1);
}

// Run the script
renameImagesInFolder().catch(console.error);
