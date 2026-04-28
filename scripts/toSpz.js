import {
    createReadStream,
    readFileSync,
    writeFileSync,
    unlinkSync
} from 'fs';
import { Readable } from 'stream';
import path from 'path';

import {
    loadPly,
    loadSpz,
    serializePly,
    serializeSpz
} from 'spz-js';

// Helper function to load either SPZ or PLY files
const loadFile = async (file) => {
    const extension = path.extname(file);
    if (extension === '.spz') {
        const fileBuffer = readFileSync(file);
        return await loadSpz(fileBuffer);
    } else if (extension === '.ply') {
        const fileStream = createReadStream(file);
        const webStream = Readable.toWeb(fileStream);
        return await loadPly(webStream);
    }
    throw new Error(`Unsupported file extension: ${extension}`);
};

const data = [
    { src: "./data/CHERRY BLOSSOMS/scene.ply", dst: "./public/CHERRY BLOSSOMS/scene.spz" },
    { src: "./data/Little-Plant/scene.ply", dst: "./public/Little-Plant/scene.spz" },
    { src: "./data/Flowers/scene.ply", dst: "./public/Flowers/scene.spz" },
];

await Promise.all(data.map((async ({ src, dst }) => {
    const gs = await loadFile(src); // or gs.spz
    const spzData = await serializeSpz(gs);
    
    writeFileSync(dst, Buffer.from(spzData));
})));
