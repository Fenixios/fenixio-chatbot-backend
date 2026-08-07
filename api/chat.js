import fs from "fs";
import path from "path";

const HF_TOKEN = process.env.HF_TOKEN;      // clave de Hugging Face (embeddings)
const GROQ_KEY = process.env.GROQ_API_KEY;  // clave del LLM (console.groq.com)
const EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2";
const GROQ_MODEL = "llama-3.3-70b-versatile"; // modelo gratuito de Groq

// Carga el índice UNA vez cuando la función arranca (no en cada petición)
const indexPath = path.join(process.cwd(), "knowledge_index.json");
const KNOWLEDGE = JSON.parse(fs.readFileSync(indexPath, "utf-8"));

function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function embedQuery(text) {
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
  const data = await resp.json();
  if (Array.isArray(data[0])) {
    const n = data.length, dim = data[0].length;
    return Array.from({ length: dim }, (_, i) =>
      data.reduce((sum, v) => sum + v[i], 0) / n
    );
  }
  return data;
}

function topChunks(queryEmbedding, k = 4) {
  return KNOWLEDGE
    .map((item) => ({ ...item, score: cosineSimilarity(queryEmbedding, item.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

async function askGroq(question, contextChunks) {
  const context = contextChunks.map((c) => `- ${c.text}`).join("\n");

  const systemPrompt = `Eres el asistente virtual oficial de Tyria, creado por los Fenixios.
Responde SOLO usando la información del CONTEXTO de abajo. Si la respuesta no está
en el contexto, di honestamente que no tienes esa información y sugiere contactar
al equipo directamente. Responde de forma breve, clara y amable, en español.

CONTEXTO:
${context}`;

  // Groq usa el mismo formato que la API de OpenAI: un arreglo de "messages"
  // con roles "system" y "user", en vez del formato "contents" de Gemini.
  const resp = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${GROQ_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: GROQ_MODEL,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: question },
      ],
    }),
  });

  if (!resp.ok) {
    console.error("Error de Groq:", resp.status, await resp.text());
    return "Lo siento, no pude generar una respuesta en este momento.";
  }

  const data = await resp.json();
  return data?.choices?.[0]?.message?.content
    ?? "Lo siento, no pude generar una respuesta en este momento.";
}

export default async function handler(req, res) {
  // CORS: permite que cualquier página llame a este endpoint
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Método no permitido" });

  try {
    const { question } = req.body;
    if (!question || question.trim().length === 0) {
      return res.status(400).json({ error: "Falta la pregunta" });
    }

    const queryEmbedding = await embedQuery(question);
    const relevant = topChunks(queryEmbedding);
    const answer = await askGroq(question, relevant);

    return res.status(200).json({ answer });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: "Error interno" });
  }
}
