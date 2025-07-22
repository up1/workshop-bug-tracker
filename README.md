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

