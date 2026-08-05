import fs from "fs";
import path from "path";

const HF_TOKEN = process.env.HF_TOKEN;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2";
const GROQ_MODEL = "llama-3.3-70b-versatile";

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
  const url = `https://router.huggingface.co/hf-inference/models/${EMBED_MODEL}/pipeline/feature-extraction`;
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${HF_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ inputs: text, options: { wait_for_model: true } }),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    console.error("Error en Hugging Face API:", errText);
    throw new Error(`HF API error: ${resp.status}`);
  }

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

  const resp = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${GROQ_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: GROQ_MODEL,
      temperature: 0.2,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: question },
      ],
    }),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    console.error("Error en Groq API:", errText);
    throw new Error(`Groq API error: ${resp.status}`);
  }

  const data = await resp.json();
  return data?.choices?.[0]?.message?.content
    ?? "Lo siento, no pude generar una respuesta en este momento.";
}

export default async function handler(req, res) {
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
    console.error("Error interno en handler:", err);
    return res.status(500).json({ error: "Error interno", details: err.message });
  }
}
