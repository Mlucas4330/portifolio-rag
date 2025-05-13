import express, { json, urlencoded } from 'express';
import cors from 'cors';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { PromptTemplate } from '@langchain/core/prompts';
import { Document } from '@langchain/core/documents';
import { collapseDocs, splitListOfDocs, } from "langchain/chains/combine_documents/reduce";

const app = express();

const HttpStatusCode = {
  OK: 200,
  BAD_REQUEST: 400,
  NOT_FOUND: 404,
  INTERNAL_SERVER: 500,
}

class HttpError extends Error {
  constructor(statusCode, message, isOperational = false) {
    super(message);
    this.statusCode = statusCode;
    this.isOperational = isOperational;
    this.name = this.constructor.name;
  }
}

app.use(cors({ origin: process.env.CORS_ORIGIN || '*' }));
app.use(json());
app.use(urlencoded({ extended: true }));

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-flash",
  temperature: 0,
  apiKey: process.env.GEMINI_API_KEY,
});

const tokenMax = 1000;

app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl} - IP: ${req.ip}`);
  next();
});

const mapPrompt = PromptTemplate.fromTemplate(`Resuma o seguinte texto de forma clara e objetiva: {context} Resumo:`);
const reducePrompt = PromptTemplate.fromTemplate(`Aqui estão vários resumos: {docs} Junte todos em um único resumo coeso:`);

async function getTokenCount(docs) {
  const counts = await Promise.all(docs.map(doc => llm.getNumTokens(doc.pageContent)));
  return counts.reduce((acc, n) => acc + n, 0);
}

async function generateSummary(content) {
  const prompt = await mapPrompt.invoke({ context: content });
  const response = await llm.invoke(prompt);
  return new Document({ pageContent: String(response.content) });
}

async function reduceSummaries(documents) {
  const input = documents.map(d => d.pageContent).join("\n\n");
  const prompt = await reducePrompt.invoke({ docs: input });
  const response = await llm.invoke(prompt);
  return String(response.content);
}

async function summarizeFromUrls(urls) {
  const rawTexts = await Promise.all(
    urls.map(async (url) => {
      const loader = new CheerioWebBaseLoader(url);
      const docs = await loader.load();
      return docs.map(d => d.pageContent).join("\n");
    })
  );

  const mappedSummaries = await Promise.all(rawTexts.map(generateSummary));

  let collapsed = mappedSummaries;
  let totalTokens = await getTokenCount(collapsed);

  while (totalTokens > tokenMax) {
    const docLists = splitListOfDocs(collapsed, getTokenCount, tokenMax);

    const newCollapsed = [];

    for (const list of docLists) {
      const collapsedDoc = await collapseDocs(list, reduceSummaries);
      newCollapsed.push(collapsedDoc);
    }

    collapsed = newCollapsed;
    totalTokens = await getTokenCount(collapsed);
  }

  const finalSummary = await reduceSummaries(collapsed);

  return finalSummary;
}

app.post('/summarizer', async (req, res) => {
  const { urls } = req.body;

  if (!urls || !Array.isArray(urls)) throw new HttpError(HttpStatusCode.BAD_REQUEST, 'Invalid URLs');

  console.log('URLs received, starting the summarizing process now.');

  const summary = await summarizeFromUrls(urls);

  console.log('Summary generated:', summary);

  res.json({ summary });
});

app.use((error, req, res, next) => {
  console.log(error);
  res.status(error.statusCode || HttpStatusCode.INTERNAL_SERVER).json({ message: error.message });
});

const PORT = process.env.PORT || '3000';
app.listen(PORT, () => console.log('Server running on port:', PORT));