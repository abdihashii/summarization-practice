import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import { OpenAI } from 'langchain/llms/openai';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { loadSummarizationChain } from 'langchain/chains';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { CheerioWebBaseLoader } from 'langchain/document_loaders/web/cheerio';
import { PromptTemplate } from 'langchain/prompts';
// Load environment variables (populate process.env from .env file)
import dotenv from 'dotenv';
dotenv.config();

const app = express();
app.use(cors());
app.use(bodyParser.json());

app.post('/summarize', async (req, res) => {
  try {
    const url = req.body.url;
    const modelName = 'gpt-3.5-turbo';

    const model = new OpenAI({
      modelName,
      temperature: 0,
    });

    // Load the website content
    const loader = new CheerioWebBaseLoader(url);
    const websiteContent = await loader.load();

    // Split the website content into documents
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 10000,
      chunkOverlap: 100,
    });
    const documents = await textSplitter.splitDocuments(websiteContent);

    for (const doc of documents) {
      const numTokens = await model.getNumTokens(doc.pageContent);
      console.log(`Number of tokens: ${numTokens}`);
    }

    const mapPrompt = `
    Write a conside summary of the following text delimited by triple =.
    Return your response in a well-formatted, multi-paragraph string.
    ==={text}===
    SUMMARY:
    `;
    const combinePromptTemplate = new PromptTemplate({
      inputVariables: ['text'],
      template: mapPrompt,
    });

    const chain = await loadSummarizationChain(model, {
      type: 'map_reduce',
      combineMapPromptTemplate: combinePromptTemplate,
    });
    const summary = await chain.call({
      input_documents: documents,
    });

    res.status(200).send(summary);
  } catch (e) {
    console.error(e);
    res.status(500).send(e);
  }
});

app.post('/summarize2', async (req, res) => {
  try {
    const url = req.body.url;

    // Load the website content
    const loader = new CheerioWebBaseLoader(url);
    const websiteContent = await loader.load();

    // Split the website content into documents
    const textSplitter = new RecursiveCharacterTextSplitter({
      separators: ['\n', '\n\n', '\t'],
      chunkSize: 10000,
      chunkOverlap: 3000,
    });
    const documents = await textSplitter.splitDocuments(websiteContent);

    const numDocuments = documents.length;

    const embeddings = new OpenAIEmbeddings({
      modelName: 'gpt-3.5-turbo',
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    const textList = documents.map((doc) => doc.pageContent);

    console.log(embeddings);

    // const vectors = await embeddings.embedDocuments(textList);

    // console.log(vectors);

    res.status(200).send({
      numDocuments,
      // vectors,
    });
  } catch (e) {
    console.error(e);
    res.status(500).send(e);
  }
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}.`);
});
