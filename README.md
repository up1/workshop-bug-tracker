## Working with Mantis
* https://github.com/xlrl/docker-mantisbt/
* https://documenter.getpostman.com/view/29959/mantis-bug-tracker-rest-api/7Lt6zkP#5cf35ffd-964b-ef90-5907-bd6b3efe87d4


## Install and configuration Mantis with Docker compose
```
$docker compose up -d mysql
$docker compose up -d mantisbt
$docker compose ps
```

Access to Mantis server
* http://localhost:8989

## Manage data in Mantis with API
* Create API token in http://localhost:8989/api_tokens_page.php

## RAG process

### 1. Get all bugs from Mantis 

Install libraries
```
$pip install -r requirements.txt
```

Run
```
$cd rag
$export API_TOKEN==<your mantis token>

$python step_1.py
```

Result in file `step_1_results.txt`

### 2. Embedding data and save to Vector database
* [Milvus](https://github.com/milvus-io/milvus)

Start Milvus in Standalone mode
```
$docker compose -f docker-compose-milvus.yml up -d
$docker compose -f docker-compose-milvus.yml ps
```

Access to UI Milvus server
* http://127.0.0.1:9091/webui/


Embedding and store data in Milvus
```
$export OPENAI_API_KEY=<your token>
$python step_2.py
```

Search bug by description (semantic search)
* https://milvus.io/docs/quickstart.md
```
$export OPENAI_API_KEY=<your token>
$python step_3.py
```

