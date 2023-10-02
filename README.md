# DocuBot Web Front-end

This is a web front-end for [DocuBot](https://github.com/bshastry/docubot), which is a Question Answering (QA) system powered by the LangChain library. It allows users to upload files in PDF, DOCX, Markdown, or plain text format, and ask questions about the content of the files. The system uses the uploaded files to create embeddings and then performs a similarity search to find the most relevant answers to the user's queries.

## Installation

To run the DocuBot web front-end, you need to install the necessary dependencies. You can install them by running the following command:

```
pip install -r requirements.txt
```

Make sure to include the `.env` file (`cp .env.template .env`) in the same directory as the Python script and assign your OpenAI API key to the `OPENAI_API_KEY` variable in the `.env` file.

## Usage

To start the web front-end, run the following command:

```
streamlit run docubot-app.py
```

Once the web front-end is running, you can access it in your web browser at `http://localhost:8501`.

## Supported File Formats

DocuBot web front-end supports the following file formats for uploading:

- PDF
- DOCX
- Markdown
- Plain text (TXT)

## Functionality

The web front-end provides the following features:

1. Uploading files: Users can upload one or more files in PDF, DOCX, Markdown, or plain text format. The files are processed and their content is used for question-answering.

2. Chunking and embedding: The uploaded files are split into text chunks using a recursive character text splitter. Each chunk is then encoded into an embedding using the OpenAI embeddings.

3. Question-answering: Users can enter a question in the provided text input field and click on the "Ask" button to get an answer. The system searches for the most relevant answers based on the input question and the embedded chunks.

4. Chat history: The web front-end displays a chat history that shows the user's questions and the corresponding answers. The chat history is updated as new questions are asked.

5. API key: Users can enter their OpenAI API key in the provided text input field to authenticate the API requests.


Please refer to the source code for more details on the implementation of each function.

## License

This code is released under the [MIT License](https://opensource.org/licenses/MIT).

## Disclaimer

This software is provided as-is. The developers make no warranties or guarantees of any kind, express or implied, about the completeness, accuracy, reliability, suitability, or availability of the software. Use of this software is at your own risk.
