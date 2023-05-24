import express, { Request, Response } from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';

import { OpenAI } from 'langchain/llms/openai';
import { loadSummarizationChain } from 'langchain/chains';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { CheerioWebBaseLoader } from 'langchain/document_loaders/web/cheerio';

// Load environment variables (populate process.env from .env file)
import * as dotenv from 'dotenv';
dotenv.config();

const app: express.Application = express();
app.use(cors());
app.use(bodyParser.json());

interface RequestBody {
  url: string;
}

app.post(
  '/summarize',
  async (req: Request<{}, {}, RequestBody>, res: Response) => {
    try {
      const { url } = req.body;

      const loader = new CheerioWebBaseLoader(url);
      const websiteContent = await loader.load();

      console.log(websiteContent);

      res.status(200).send(websiteContent);
    } catch (e) {
      console.error(e);
      res.status(500).send(e);
    }
  },
);

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}.`);
});
