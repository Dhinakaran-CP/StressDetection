const fs = require('fs');
const path = require('path');
const srcDir = path.join(process.cwd(), 'frontend', 'src');

function checkFile(file) {
    const content = fs.readFileSync(file, 'utf8');
    const regex = /import\s+.*?\s+from\s+['"](.*?)['"]/g;
    let match;
    let hasError = false;
    while ((match = regex.exec(content)) !== null) {
        const importPath = match[1];
        if (importPath.startsWith('.')) {
            // resolve relative path
            const resolvedPath = path.resolve(path.dirname(file), importPath);
            // check if file exists with .js, .jsx, .ts, .tsx, .css, .svg, .png etc
            const exts = ['', '.js', '.jsx', '.ts', '.tsx', '.css', '.svg'];
            let exists = false;
            for (let ext of exts) {
                if (fs.existsSync(resolvedPath + ext)) {
                    exists = true;
                    break;
                }
                if (fs.existsSync(path.join(resolvedPath, 'index' + ext)) && ext !== '') {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                console.log('Broken import in ' + file + ': ' + importPath);
                hasError = true;
            }
        }
    }
}

function walkDir(dir) {
    if (!fs.existsSync(dir)) return;
    const files = fs.readdirSync(dir);
    for (let file of files) {
        const fullPath = path.join(dir, file);
        if (fs.statSync(fullPath).isDirectory()) {
            walkDir(fullPath);
        } else if (fullPath.endsWith('.js') || fullPath.endsWith('.jsx')) {
            checkFile(fullPath);
        }
    }
}

walkDir(srcDir);
console.log('Frontend import check done.');
