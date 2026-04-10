import fs from 'fs';
import path from 'path';

const file = path.join(process.cwd(), 'node_modules', '@assistant-ui', 'react', 'dist', 'index.d.ts');
if (fs.existsSync(file)) {
  const content = fs.readFileSync(file, 'utf8');
  const match = content.match(/ExternalStoreRuntimeProps[^}]*}/s);
  if (match) {
    console.log("Found in index.d.ts:");
    console.log(match[0]);
  } else {
    // maybe it's in another file
    for (const filename of fs.readdirSync(path.join(process.cwd(), 'node_modules', '@assistant-ui', 'react', 'dist'))) {
      if (filename.endsWith('.d.ts')) {
        const fileContent = fs.readFileSync(path.join(process.cwd(), 'node_modules', '@assistant-ui', 'react', 'dist', filename), 'utf8');
        const match2 = fileContent.match(/ExternalStoreRuntimeProps[^}]*}/s);
        if (match2) {
            console.log(`Found in ${filename}:`);
            console.log(match2[0]);
            break;
        }
      }
    }
  }
} else {
  console.log("Not found.");
}
