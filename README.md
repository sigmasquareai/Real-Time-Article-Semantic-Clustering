


# Real Time Text Analytics
This backend ML service deploys Transformer Model for upstream NLP task, Semantic Similarity, with Promethes stack for monitoring. 

The ML Service is containerized REST API developed with FastAPI and ML client service is python script, running the whole pipeline of getting articles from database, calculating embeddings, finding similar embddings and writing similarity results to DB after filtering against a set threshold.

Tested on multiple configurations including

 1. CPU vs GPU (current deployment supports CPU only)
 2. REST API vs in-script model loading

## Stack

 - **ML-Server** *(service)* 
		 - Sentence-Transformers *(python framework)*
		 - FastAPI *(REST API)* 
 - **ML-Client** *(service)*
		 - SQLalchemy *(postgresql client)*
		 - Milvus *(client)*
 - **Milvus** *(service)*
		 Dense Vector (embeddings) caching server, configurations at `milvus/conf/server_config.yaml`
		 Configuration params: https://milvus.io/docs/v1.0.0/milvus_config.md
 - **Prometheus** *(service)* 
		 Timeseries database for logs and event storage, configurations at `promstack/prometheus/prometheus.yml` alert rules at `promstack/prometheus/alert_rules.yml`
 - **Pushgateway** *(service)* 
		Standalone listening server to receive incoming event logs.
 - **AlertManager** *(service)* 
		 Manages alert publications, configurations at `promstack/alertmanager/alertmanager.yml`
 - **Grafana** *(service)*
		 Dashboard for real time stats and analysis, configuration steps at: https://milvus.io/docs/v1.0.0/setup_grafana.md and dashboard config at `promstack/grafana/dashboard.json`

## Setup
Run `bash ./provision.sh` 

This shell script installs docker and docker-compose on local machine if not already exists and initiates all the services. It takes following positional arguments:

1. `buildmilvus` build cache the embeddings of last 03 days *(must be provided for first time deployment)*
2. `testml` this will test the ML server response

### Debug
In the project root directory, following commands will stop or resume all the services.

Stop all the services  `sudo docker-compose stop`
Shut all the services  `sudo docker-compose down`
Restart is required if any changes are made in configs  `sudo docker-compose restart`
Start all the services in background  `sudo docker-compose up -d` but to view real-time console logs, run the command without `-d` flag.
## Monitoring
Make sure the respective ports are opened for each service.

**Grafana** `http://<ip>:3000`
**Prometheus** `http://<ip>:9090`
**Pushgateway** `http://<ip>:9091`
**Alertmanager** `http://<ip>:9093`
