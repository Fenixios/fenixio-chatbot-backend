import fs from "fs";
import path from "path";

const HF_TOKEN = process.env.HF_TOKEN;
const EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2";

if (!HF_TOKEN) {
  console.error(
    "Falta la variable de entorno HF_TOKEN. Agrégala en Vercel → Settings → Environment Variables (asegúrate de que esté marcada para 'Build' y no solo 'Runtime')."
  );
  process.exit(1);
}

function loadChunks(filePath) {
  const text = fs.readFileSync(filePath, "utf-8");
  return text
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .filter(Boolean);
}

async function embed(text) {
  const resp = await fetch(
    `https://router.huggingface.co/hf-inference/models/${EMBED_MODEL}/pipeline/feature-extraction`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ inputs: text, options: { wait_for_model: true } }),
    }
  );

  if (!resp.ok) {
    throw new Error(`Hugging Face respondió con error ${resp.status}: ${await resp.text()}`);
  }

  const data = await resp.json();

  if (Array.isArray(data[0])) {
    const n = data.length;
    const dim = data[0].length;
    return Array.from({ length: dim }, (_, i) => data.reduce((sum, v) => sum + v[i], 0) / n);
  }
  return data;
}

async function main() {
  const knowledgePath = path.join(process.cwd(), "knowledge.txt");
  const chunks = loadChunks(knowledgePath);

  console.log(`Generando embeddings para ${chunks.length} fragmentos...`);

  const index = [];
  for (let i = 0; i < chunks.length; i++) {
    console.log(`  Procesando fragmento ${i + 1}/${chunks.length}...`);
    const embedding = await embed(chunks[i]);
    index.push({ id: i, text: chunks[i], embedding });
  }

  const outPath = path.join(process.cwd(), "knowledge_index.json");
  fs.writeFileSync(outPath, JSON.stringify(index), "utf-8");

  console.log(`Listo: ${index.length} fragmentos indexados en knowledge_index.json`);
}

main().catch((err) => {
  console.error("Error generando el índice de conocimiento:", err);
  process.exit(1); 
});
