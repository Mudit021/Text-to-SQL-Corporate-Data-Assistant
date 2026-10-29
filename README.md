# Corporate Data Assistant

A local, privacy-first Streamlit chatbot that translates business questions into
read-only SQL over a SQLite database. It defaults to
`company_database.sqlite` (employees, departments, projects, payroll, and
attendance) and uses **Falcon-H1 0.5B Instruct Q6_K**, which is about 430 MB in
Ollama. All inference and data access stay on the machine.

## How it works

A 0.5B model cannot reliably follow LangChain's multi-step ReAct agent format, so
the app uses a simpler pipeline:

1. Select the tables most relevant to the question from the real schema.
2. Generate one SQLite `SELECT` using schema-aware hints and examples.
3. Validate and execute it through the read-only database wrapper.
4. On failure, retry with the SQLite error and the correct available columns.
5. Normalize common MySQL/SQL Server syntax mistakes to SQLite before validation.
6. Send only the relevant examples, plus the exact stored values of filter
   columns (department names, statuses, skills), so case-sensitive SQLite
   comparisons match the data.
7. Add bridge tables from the foreign-key graph, so department questions about
   payroll or reviews join through `employees` instead of inventing a key.
8. Check the generated SQL for semantic mistakes that still execute (an age
   computed from a birth year, a department name compared to `job_title`, a
   global `MAX(review_id)` used for each employee's latest review) and repair it.
9. If the requested field does not exist, explain which fields are available
   instead of returning a generic failure.
10. Summarize the actual returned rows deterministically, so the model cannot
    invent company names or numbers in the final answer. Empty date-filtered
    results state the date range actually present in the table.

## 1. Prerequisites

- Python 3.12 (the pinned dependencies are tested on Python 3.12).
- [Ollama](https://ollama.com/download) installed and running.
- The database file. This project currently uses
  `company_database.sqlite` (or `company_database.db`).

## 2. Install the application

From this project directory:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe` directly as
shown below.

## 3. Import the Falcon-H1 model

Download the Q6_K GGUF file (410 MB) from Hugging Face:

```powershell
New-Item -ItemType Directory -Force models | Out-Null
curl.exe -L --fail -o models\Falcon-H1-0.5B-Instruct-Q6_K.gguf `
  https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct-GGUF/resolve/main/Falcon-H1-0.5B-Instruct-Q6_K.gguf
```

Create the Ollama model using the included Modelfile:

```powershell
ollama create falcon-h1-0.5b:q6k -f Modelfile.falcon-h1
ollama run falcon-h1-0.5b:q6k "Reply with OK"
```

The imported model is about 430 MB, under the 500 MB limit. Start the Ollama
server with `ollama serve` if it is not already running.

The model uses the Falcon-LLM License; review the model card before
redistribution.

To use a different installed model, enter its name in the app sidebar or set:

```powershell
$env:OLLAMA_MODEL = "falcon-h1-0.5b:q6k"
```

`OLLAMA_BASE_URL` can override the default `http://localhost:11434`.

## 4. Database

The app automatically chooses the first of these files that exists:

1. `company_database.sqlite`
2. `company_database.db`
3. `mca_master.db`

You can also enter any SQLite path in the sidebar. The app reads the real schema
at startup and does not require the mock MCA schema. It opens the file strictly
read-only.

## 5. Launch

```powershell
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Using `.venv\Scripts\python.exe` guarantees Streamlit runs inside the project
environment even if activation was skipped. Open <http://localhost:8501>.

If startup reports `ModuleNotFoundError: No module named 'langchain_community'`,
the app was launched with the global Python. Re-run it with the command above.

Ask questions such as:

- How many employees are in the company?
- How many active employees are there?
- List the top 5 highest-paid employees.
- What is the average salary by department?
- How many employees were hired last year?
- How many employees were absent today?
- What is the total net pay by pay month?
- Which projects are delayed?
- How many employees work remotely?
- How many male and female employees are there?

The first question initializes the pipeline. The SQLite engine and Ollama client
are then retained in `st.session_state`; later messages do not rebuild them
unless you change the database path, model, or Ollama URL. Use **Clear chat** to
clear the transcript without dropping the cached pipeline.

## Security and resource controls

- **Strict read-only SQLite:** connections use a `file:...?mode=ro` URI and also
  execute `PRAGMA query_only=ON`. Writes fail at the database layer.
- **SELECT-only guard:** sqlglot parses every generated statement. Only one
  SELECT, CTE, or set-operation statement is accepted; INSERT, UPDATE, DELETE,
  DDL, PRAGMA, ATTACH, and multiple statements are rejected.
- **Hard row cap:** each statement is wrapped as
  `SELECT * FROM (...) LIMIT 15`, even if the model writes a larger LIMIT.
  Individual text cells are truncated at 300 characters, and the result string
  is capped at 12,000 characters.
- **No string-built SQL from user input:** the question is sent to the model;
  only generated SQL is executed, and only after validation.
- **Automatic repair:** SQLite errors include the failed SQL, available columns,
  and explicit SQLite syntax hints. The app makes up to two repair attempts, then
  gives a schema explanation when a field truly does not exist. Semantic checks
  also reject queries that run but answer a different question.
- **Faithful summaries:** final text is rendered from the returned rows, not
  generated by the 0.5B model, preventing hallucinated values.
- **Small context:** only relevant tables are sent to the model, with no sample
  rows, a 4096-token context, and a 220-token generation cap.

## Limitations

This is a local single-user tool with no authentication, so keep Streamlit bound
to `127.0.0.1`. A 0.5B model handles common single-table and simple join
questions well, but very complex or ambiguous questions may still require
rephrasing or a larger model. The 15-row cap means exhaustive "list every row"
requests should be expressed as aggregates or top-N queries.
