import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import { OpenAI, OpenAIChat } from 'langchain/llms/openai';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { loadSummarizationChain } from 'langchain/chains';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { CheerioWebBaseLoader } from 'langchain/document_loaders/web/cheerio';
import { PromptTemplate } from 'langchain/prompts';
import { Document } from 'langchain/document';
import skmeans from 'skmeans';
import tf from '@tensorflow/tfjs-node';
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
    // GPT-3.5 Turbo model
    const llm3 = new OpenAIChat({
      modelName: 'gpt-3.5-turbo',
      maxTokens: 4097,
      openAIApiKey: process.env.OPENAI_API_KEY,
      // streaming: true,
    });
    // GPT-4 model
    const llm4 = new OpenAIChat({
      modelName: 'gpt-4',
      maxTokens: 4097,
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    // Load the website content
    const loader = new CheerioWebBaseLoader(url);
    const websiteContent = await loader.load();

    // Split the website content into documents
    const textSplitter = new RecursiveCharacterTextSplitter({
      separators: ['\n', '\n\n', '\t'],
      chunkSize: 10000,
      chunkOverlap: 500,
    });
    const documents = await textSplitter.splitDocuments(websiteContent);

    // Get the number of documents that this website was split into
    const numDocuments = documents.length;

    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    // Create an array of the text from each document
    const textList = documents.map((doc) => doc.pageContent);

    // Embed the text and store the vectors
    const vectors = await embeddings.embedDocuments(textList);

    // Cluster the vectors using k-means
    const kMeans = skmeans(vectors, 10);

    // Create an array of the indices of the closest points to each cluster center
    const closestIndices = [];
    for (let i = 0; i < kMeans.centroids.length; i++) {
      // Get the list of distances from that particular cluster center
      const vectorsTensor = tf.tensor(vectors);
      const difference = vectorsTensor.sub(kMeans.centroids[i]);
      const distances = difference.norm(2, -1).arraySync();

      // Find the list position of the closest point
      const closestIndex = distances.indexOf(Math.min(...distances));

      // Append that position to your closest indices list
      closestIndices.push(closestIndex);
    }

    // Sort the closes indices so that they are in order
    const selectedIndices = closestIndices.sort((a, b) => a - b);

    // Initialize a map_reduce summary chain using gpt-3.5-turbo
    const summaryChain = loadSummarizationChain(llm3, {
      type: 'map_reduce',
    });

    // Map the selected indices to the original website documents
    const selectedDocs = selectedIndices.map((doc) => {
      return documents[doc];
    });

    // Create a list of summaries
    const summaryList = [];
    for (let i = 0; i < selectedDocs.length; i++) {
      const doc = selectedDocs[i];

      const chunkSummary = await summaryChain.call({
        input_documents: [doc],
      });

      summaryList.push(chunkSummary.text);

      console.log(
        `Summary #${i + 1}: (chunk #${
          selectedIndices[i]
        } - Preview: ${chunkSummary.text.slice(0, 250)}) \n\n`,
      );
    }

    // Join the summaries together
    const summaries = summaryList.join('\n\n');

    // Convert it back to a document
    const summaryDocument = new Document({
      pageContent: summaries,
    });

    // Get the number of tokens of the total summary
    const summaryNumTokes = await llm3.getNumTokens(
      summaryDocument.pageContent,
    );

    // Initialize a stuff summary chain using gpt-4
    const stuffChain = loadSummarizationChain(llm4, {
      type: 'stuff',
    });

    // Summarize the combined summaries into a final summary using gpt-4
    const finalSummary = await stuffChain.call({
      prompt: `
      You will be given a series of summaries from a website. The summaries will be enclosed in triple equals (===).
      Your task is to give a verbose summary of the website's content.
      The reader should be able to understand the website's content from your summary alone.
      ==={text}===
      VERBOSE SUMMARY:
      `,
      input_documents: [summaryDocument],
    });

    res.status(200).send({
      numDocuments,
      numTokens: summaryNumTokes,
      finalSummary: finalSummary.text,
    });
  } catch (e) {
    console.error(e.stack);
    res.status(500).send(e);
  }
});

const PORT = process.env.PORT || 8080;

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}.`);
});
