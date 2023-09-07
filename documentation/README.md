# GITS - a machine learning platform
## Architecture overview

The following describes the second, extended version of the GITS AI/ML processing platform (the first version can be found [here](https://github.com/VicomtechV3/GITS/tree/main)). Apart from allowing for assigning different ML models to different workers/services performing an ML-related task, this version has additional **monitoring and model retraining** modules.

> *For more information, please check the [GITS video](https://vicomtech.box.com/s/su5awommp6wmymzcq4873mjqg9tscbn1) (39:26).*

The extended GITS architecture consists of three parts:

- I) a Python *Django* server with its *admin UI* and additional *dashboards*. A simpe SQLite database is used to store Django user data.
- II) the inference pipeline (essentially the previous GITS architecture) consisting of the *queue_manager*, a *RabbitMQ broker* and a set of *workers*, i.e. modules/services performing ML-related tasks such as regression or classification on the data coming in from the *data_source*. A MongoDB database is used to store (meta-) data related to the jobs processed by the workers.
- III) the new monitoring and retraining pipeline consisting of a TimescaleDB database and an associated *ml_data_api*, a *ground_truth_collector*, which constantly adds new training data to the database, a *monitoring_api* to calculate ML model drift and other relevant metrics and a *retraining_api* that allows for online retraining of ML models based on newly available training data.

For simplicity's sake, parts II and III are provided here in one Docker container, whereas part I runs in its own container - the idea being that parts II and III can run independently from Django (part I). The queue_manager, ml_data_api, monitoring_api and retraining_api all have their own endpoints allowing to access their functionalities as documented in the given [Postman collection](gits.postman_collection.json). Refer to the image below for a full overview of the GITS architecture.

![Gits architecture](images/GITS_MLOps_architecture_v4.png)

## Module overview

| Part | Module | Technology | Main functionalities | URLs |
|---|---|---|---|---|
| I | Django server | [Django (Python)](https://www.djangoproject.com/) | User management; defining service (ML worker) information and uploading ML models; accessing dashboards | http://localhost:8000/, http://localhost:8000/admin, http://localhost:8000/dash_main_dashboard |
| I | SQLite DB | [sqlite](https://www.sqlite.org/index.html) | Storing Django user data as well as service and ML model data | |
| II | data_source | [aio_pika](https://aio-pika.readthedocs.io/en/latest/) | Data source sending new production data in a defined frequency (here set to sending 1 message every 2 seconds) | |
| II | RabbitMQ | [RabbitMQ](https://www.rabbitmq.com/) docker | Message broker distributing messages sent from the data_source to connected workers | http://localhost:15672/#/ (user: vicomtech, PW: vicomGits2021) |
| II | worker / service | [aio_pika](https://aio-pika.readthedocs.io/en/latest/), [pycaret](https://pycaret.gitbook.io/docs/) | Core ML worker performing ML tasks such as regression, classification etc. on the incoming data. One can launch as many worker instances as necessary in order to process messages coming in from the data_source in parallel | |
| II |queue_manager | [fastapi](https://fastapi.tiangolo.com/), [mongoengine](http://docs.mongoengine.org/) | Core API allowing to perform CRUD operations on job data, and on service/worker information and ML model files | http://localhost:8001/docs#/ |
| II | MongoDB database | [MongoDB](https://www.mongodb.com/) docker | Storing data related to every ML job processed by workers; storing meta data such as processing times, job status etc.|
| III | ground_truth_collector | [pandas](https://pandas.pydata.org/) | Sending new ground truth (labeled) data to the training database in defined intervals | |
| III | ml_data_api | [fastapi](https://fastapi.tiangolo.com/), [psycopg2](https://www.psycopg.org/docs/) | Allowing CRUD operations for all ML-related training, testing and production data to be stored in TimescaleDB | http://localhost:8004/docs#/ |
| III | TimescaleDB | postgresql, [timescale](https://www.timescale.com/) | Storing tabular ML data (i.e. features, labels and timestamps) | http://localhost:9000/browser/ ([pgadmin](https://www.pgadmin.org/) interface; user: admin@admin.com, PW: admin); follow https://mccarthysean.dev/001-05-timescale for setup; adding a new DB server by righ-clicking `Servers -> Register -> Server` and then entering the following information in the respective tabs (again PW: admin): ![Timescale general tab](images/timescale_general.png) and ![Timescale connection tab](images/timescale_connection.png) then expand in the pgadmin tool `TimescaleDB Local -> Databases -> postgres -> Schemas -> public -> Tables` to see the created data tables containing the ML data uploaded by the ml_data_loader and the ground_truth_collector.|
| III | monitoring_api | [fastapi](https://fastapi.tiangolo.com/), [evidentlyai](https://www.evidentlyai.com/) | Allowing to perform data quality, data drift, target drift and regression performance analyses provided by the evidentlyai library; outputs can be in JSON and HTML format | http://localhost:8005/docs#/ |
| III | retraining_api | [fastapi](https://fastapi.tiangolo.com/), [pycaret](https://pycaret.gitbook.io/docs/), [explainerdashboard](https://explainerdashboard.readthedocs.io/en/latest/) | Retraining ML models using the data stored in TimescaleDB (e.g. using new training data as provided by the ground_truth_collector); calculating model performance on data stored in TimescaleDB; creating an explainer dashboard for feature importance analysis etc. as provided by the explainerdashboard library (beta) | http://localhost:8006/docs#/ |

### Additional modules provided in the repository

| Part | Module | Technology | Main functionality | URLs |
|---|---|---|---|---|
| - | ml_data_loader | [psycopg2](https://www.psycopg.org/docs/), [pandas](https://pandas.pydata.org/) | Uploading initial labeled training and testing data from the use case dataset to TimescaleDB; service is executed once upon running `docker-compose up` and then stopped | - |
| - | pycaret_files | [pycaret](https://pycaret.gitbook.io/docs/), [pandas](https://pandas.pydata.org/) | Jupyter Notebook used to train initial ML models using PyCaret  | - |


## Use case
To demonstrate the ML capabilities of the GITS architecture, we implemented a use case based on publicly available data, to be found [here](https://www.kaggle.com/datasets/podsyp/production-quality?select=data_X.csv). The use case's objective is to predict the product quality produced by a roasting machine based on the machine's operating parameters such as temperature data collected by sensors. The input features ("X") and the ground truth quality data (labels, "Y") can be downloaded from the provided Kaggle website or from the links given below. We consider the problem to be a __regression__ problem and hence will use the respective regression functions provided by PyCaret to train and test ML models. You will find a Jupyter notebook used to create ML models based on the given data [here](pycaret_files/GITS_PyCaret_KaggleData.ipynb).

> NOTE: Follow these links to download the input datasets [`data_X.csv`](https://vicomtech.box.com/s/gmwty1qvityvca7ued47zvlrtj8ficyj) and [`data_Y.csv`](https://vicomtech.box.com/s/0jhqbolp4gx1v7ruflmr8gn487lj4wtz) as used in the provided [PyCaret Jupyter notebook]((pycaret_files/GITS_PyCaret_KaggleData.ipynb)).

Fur further details on how train, test and evaluate ML models with PyCaret, please refer to the PyCaret documentation:

- https://pycaret.gitbook.io/docs/get-started/quickstart#regression
- https://pycaret.gitbook.io/docs/get-started/tutorials
- https://pycaret.gitbook.io/docs/learn-pycaret/examples

> NOTE: The entire GITS platform has been designed around the mentioned use case. Therefore, there are several hardcoded parts of the code relating to the given data structure, for example. If you want to use the GITS architecture for a different ML task using a different data set, adaptations need to be made.


## GITS flow of information

1) The data_producer sends some input data directly to a RabbitMQ queue, which is consumed by workers listening/scubscribed to this queue. Typically, this input data contains features based on which the workers should perform an ML task such as regression, classification etc.
2) Whichever worker is free (i.e. not busy processing another incoming message), receives the incoming data and firstly creates a new job by making a POST request to the queue_manager. The worker then checks which ML model it is supposed to apply to the data sent by the data_source. This information is provided by the queue manager using the unique service/worker name that is provided in the [worker's .env file](worker_product_quality_a/.env) (parameter `SERVICE_NAME`). The respective URL may be http://localhost:8001/queue_manager/get_services?service_name=service_product_quality_A, for example. 
3) For this information to be defined, the user first needs to login to Django admin (http://localhost:8000/admin) and a) upload a ML model and its PyCaret configuration file and b) create the service/worker and associate it with the desired ML model. Changes in Django admin are sent to the queue_manager, which stores this information in the MongoDB.

> NOTE: Changes in Django (i.e. POSTing data to create or delete services etc.) are reflected in the queue_manager and MongoDB, but not vice versa.

4) Knowing the ML model it is supposed to use, the worker now can process the incoming message, apply the model and perform, for example, a regression task on the data. Once this is done, results are stored in TimescaleDB via a call to the ml_data_api (e.g. http://localhost:8004/ml_data_api/insert_data/) in the "production" data table.
5) Finally, the worker updates the status of the current job by calling respective queue_manager endpoints (check [worker.py](worker_product_quality_a/worker.py) and [ml_model.py](worker_product_quality_a/ml_model.py) for more details). A finished job may look like this:

        # job object as stored in MongoDB

        {
        "_id": {
                "$oid": "63bfce0a0961891b08de58a8"
        },
        "payload": {
                "product_quality": [
                169.94,
                279.0,
                256.0,
                256.0,
                321.0,
                469.0,
                574.0,
                535.0,
                259.0,
                258.0,
                250.0
                ]
        },
        "last_status": "finished",
        "result": [
                {
                "h_data": 169.94,
                "t_data_1_1": 279.0,
                "t_data_1_2": 256.0,
                "t_data_1_3": 256.0,
                "t_data_2_3": 321.0,
                "t_data_3_1": 469.0,
                "t_data_3_2": 574.0,
                "t_data_3_3": 535.0,
                "t_data_5_1": 259.0,
                "t_data_5_2": 258.0,
                "t_data_5_3": 250.0,
                "quality": 366.89508898945127,
                "date_time": "2023-01-12T09:08:26.590699"
                }
        ],
        "message": "Results calculated successfully.",
        "metadata": {
                "start_date": "2023-01-12 09:08:26.293616",
                "end_date": "2023-01-12 09:08:26.596140",
                "time_elapsed_sec": "0.302524",
                "routing_key": "gits_product_quality_A",
                "worker": "service_product_quality_A"
        }
        }

6) Results, job status, amount of data in the databases etc. can be visualized via simple dashboards made with [Plotly Dash](https://dash.plotly.com) (e.g. http://localhost:8000/dash_main_dashboard) that are hosted on the Django server:

![Main dashboard](images/dash_main.png)

7) As the workers keep adding ML results (i.e. production data) to TimescaleDB, the monitoring_api can be used to monitor, for example, production **data quality, data drift, target drift or regression performance** compared to the training and testing data stored in TimescaleDB. Furthermore, training data keeps growing as the ground_truth_collector keeps adding new labeled data to the training table in TimescaleDB. The monitoring_api provides endpoints that calculate data/model drift and relevant parameters using the [EvidentlyAI](https://www.evidentlyai.com/) library. Results can be provided in JSON format or as HTML (i.e. preset dashboards as provided out-of-the-box by EvidentlyAI). See, for example, http://localhost:8000/dash_monitoring_links to generate such HTMLs. The dashboard at http://localhost:8000/dash_monitoring uses the monitoring_api to calculate drift and performance metrics for real-time monitoring.
![Monitoring dashboard 1](images/dash_target_drift.png)

![Monitoring dashboard 2](images/dash_real_time.png)

> NOTE: The provided dashboards are only simple example dashboards of possible UIs and may show low performance or errors compared to traditional front-ends based on JS.

8) The ground_truth_collector keeps adding new training data and at the same time production data is growing and may show data or target drift. Hence, retraining the current model may be beneficial. For this, the retraining_api provides endpoints to analyse model performance on training, testing and production data stored in TimescaleDB and in particular provides one endpoint that triggers retraining based on the current training data leveraging [PyCaret](https://pycaret.gitbook.io/docs/get-started/quickstart) functions. 

> NOTE: Retraining might require a long time (for the given use case several minutes), which is why the current implementation can be considered a "beta" version, still suffering from server timeouts. Calling the endpoint via Postman, however, should work and yields retraining results, comparing the newly created model to a previous model.

        # example retraining results:

        {
        "retrained_model_results": {
                "Model": "CatBoost Regressor",
                "MAE": 10.3334,
                "MSE": 195.649,
                "RMSE": 13.9875,
                "R2": 0.9076,
                "RMSLE": 0.0373,
                "MAPE": 0.0267
        },
        "previous_model_results": {
                "Model": "CatBoost Regressor",
                "MAE": 10.2981,
                "MSE": 193.1445,
                "RMSE": 13.8976,
                "R2": 0.9088,
                "RMSLE": 0.037,
                "MAPE": 0.0266
        }
        }

9. The retraining_api provides one additional endpoint to generate a *static* explainerdashboard using the Python library carrying the same name (URL example: http://localhost:8006/retraining_api/get_explainer_dashboard?model_name=gits_kaggle_2&config_name=config_gits_kaggle_2&evaluate_on=production). It permits calculating, for example, feature importance scores etc. for a model applied to a specific dataset. Results are returned as a *static* HTML (not interactive as in Jupyter!). 

> NOTE: This is a beta version of an explainerdashboard implementation as calculating the dashboard inside a web service can take very long time (for the example data more than half an hour!). This is much faster and easier in a Jupyter notebook, see the [ML model training notebook using Pycaret](pycaret_files/GITS_PyCaret_KaggleData.ipynb) for examples.


# Prerequisites
- Install [Docker](https://docs.docker.com/get-docker/) and [docker-compose](https://docs.docker.com/compose/install/)
- Note that all services have individual `.env` files to define critical configuration parameters. These files typically contain sensitive information and therfore should ideally be removed from source control (e.g. via updating the [.gitignore](.gitignore) file). Configuration parameters from `.env` files are ingested via [python-decouple](https://pypi.org/project/python-decouple/) in Python scripts.

# Setup with Docker

## 1) Generate Docker network
Generate a common Docker network on which both the Django server (in its separate container) and the GITS services run to communicate with each other. Example network name: `gits-network`:

        docker network create gits-network

> NOTE: If you want to change the network name you need to update [Django's docker-compose](django-dashboard-gradientable-master/docker-compose.yml) and the [GITS docker-compose](docker-compose.yml) file.

## 2) Start the Django server
Start the Django server via docker-compose (first time with `--build`). You need to be in the `django-dashboard-gradientable-master` directory:

        cd django-dashboard-gradientable-master
        docker-compose up --build -d

Make sure the Django server is running by accessing http://localhost:8000/admin. Login with user `admin` and password `admin`. 

> NOTE: This super user is created automatically when running [Django's Dockerfile](django-dashboard-gradientable-master/docker-compose.yml) with the `RUN echo "from django.contrib.auth...` command. WARNING: In a production environment, login as `admin` and change immediately the password to something that is more secure. Further, you should probably change `DEBUG` to `False` and change the `SECRET_KEY` parameter in [Django's .env file](django-dashboard-gradientable-master/.env)

## 2) Start the GITS services
Start all GITS services via docker-compose (first time with `--build`), being in the main GITS directory:

        docker-compose up --build -d

Make sure all services are up and running. Check first http://localhost:8001/ to see whether the queue manager is running. Output should be:

        {"message":"This is the queue manager."}

If there are any issues, please check the Docker logs.

> NOTE: The service `gits_ml_data_loader` will only be run once upon calling docker-compose up and will then stop. This is because this service only pre-populates the "training" and "testing" data tables in TimescaleDB by uploading the provided two dataframes df_1.csv and df_4_5.csv. Those dataframes represent the chosen training and testing data extracted from the original data_X and data_Y dataframes given in the [Kaggle dataset](https://www.kaggle.com/datasets/podsyp/production-quality?select=data_X.csv). You can see how all relevant dataframes were created in the [Jupyter notebook used for training](pycaret_files/GITS_PyCaret_KaggleData.ipynb).


## 4) A note on the data_producer and ground_truth_collector
These two services are here included in the [main docker-compose file](docker-compose.yml) and are started immediately, as soon as all required services are up and running (this may take a while). If you want to change message intervals and/or the RabbitMQ queues to which data should be sent, check the respective .env files for these services:

- data_producer: [.env](data_source_product_quality/.env)
- ground_truth_collector: [.env](ground_truth_collector/.env)

Further, queues need to match the queues the workers are listening to, as defined in the [worker's .env file](worker_product_quality_a/.env).

## 5) Check the data visualization
Data is being sent to the Rabbit broker by the data_producer to the queue, which is consumed by the worker. This worker applies its associated ML model to the data and stores the results both in MongoDB and in TimescaleDB. Results are exposed by the queue_manager and consumed by a [Plotly dashboard](http://localhost:8000/dash_main_dashboard) in Django as shown above.

## 6) Uploading an ML model and associating it with a service/worker
When accessing the dashboard for the first time (allow a good minute for all services to be up and running), one will note that messages are being processed, but with errors (see also docker logs for more details). This is due to the fact that the worker cannot retrieve the information which ML model it is supposed to use from the queue_manager. In order to solve this, we first need to add a model to the queue_manager and define which worker/service is supposed to use this model. For convenience, one ML model and its configuration file have already been added to the [`pycaret_files` folder](pycaret_files) as .pkl files (they have been generated using the [Jupyter notebook mentioned before](pycaret_files/GITS_PyCaret_KaggleData.ipynb)). Upload this model and create a service matching the data from the [worker's .env file](worker_product_quality_a/.env) by entering Django's admin section (http://localhost:8000/admin/) and log in as admin. Click on "Ml algorithms" and in the upper right corner "Add ML Algorithm". Then enter the model's details and upload the file.

> NOTE: The algorithm name is a model's unique identifier.

Select one of the pickled algorithms to be uploaded and click "Save" to upload.

> NOTE: The algorithm is uploaded to the Django server (to be found in `django-dashboard-gradientable-master/uploaded_files/ml_models`) and Django's sqlite database as well as to the MongoDB via the queue manager. To check the MongoDB entry, go to http://localhost:8001/queue_manager/get_algorithms.
*Being both in Django's sqlite and MongoDB, mismatches may arise in case one of the services needs to be re-initiated without persisting the databases (e.g. via docker-compose down)!* Here, Django admin was used for convenience only - ideally an independent front-end would allow uploading algorithms via the queue manager's endpoints to the MongoDB only.

Once the algorithm has been successfully uploaded, move one level up to "Services" and click "Add Service". The service name you enter needs to match the `SERVICE_NAME` parameter defined for a specific direct worker (see the [worker's .env file](worker_product_quality_a/.env)). Here, this name is `service_product_quality_A`.

> NOTE: The service name is its unique identifier.

As "Associated algorithm" choose the model we just uploaded previously. You can check the available services and their associated algorithms by accessing http://localhost:8001/queue_manager/get_services.

Now we are ready to send data to the worker from the data producer. When you return to the main dashboard, you will see that the number of finished jobs will increase as now the workers should not have any errors anymore. All services should now be running correctly and you can check the other dashboards and Postman endpoints to test the entire GITS system.

> NOTE: You can control the number of workers in the [general docker-compose file](docker-compose.yml) with the

        deploy:
        mode: replicated
        replicas: 2

parameters for `gits_worker_product_quality_a`.


# Standalone mode for local development
For local development it may be more convenient to run all services on the local machine, without using Docker. For this, the following changes need to be made.
## 1) Launch Django in standalone mode
To launch Django in standalone mode, change the `DOCKER_SETUP` parameter in [Django's .env file](django-dashboard-gradientable-master/docker-compose.yml) to `standalone`:

        DOCKER_SETUP=standalone

Then, create and activate a virtual environment using the `requirements.txt` file in the `django-dashboard-gradientable-master` directory.

        cd django-dashboard-gradientable-master
        pip install -r requirements.txt

Further, while being in the `django-dashboard-gradientable-master` directory, run the following commands to set up the Django development server on your machine:

        python manage.py makemigrations
        python manage.py migrate
        python manage.py createsuperuser  # use e.g. admin/admin

        # start the server with
        python manage.py runserver

Django will be running on http://localhost:8000 as for the Docker setup.

## 2) Launch all other services in standalone mode
First, comment all services except `rabbitmq-server`, `mongo`, `pgadmin` and `timescale` in the general [docker-compose](docker-compose.yml) file. The `networks` field should also be commented.

In the main directory, run

        docker-compose up -d

to launch the RabbitMQ, MongoDB and TimescaleDB servers (make sure to stop all previously created containers). Then, create virtual environments for each of the services (except Django) using the `requirements.txt` files in the respective folders. Further, change all `DOCKER_SETUP` parameters to `standalone` (as previously for Django), in the respective .env files for each service. After that, open several terminals (one for each service) and activate the virtual environments. Then (from their respective directories!)

- launch the queue_manager with 

        python run_queue_manager.py

- launch the worker_product_quality_a with

        python worker.py

- launch the ml_data_api with

        python run_ml_data_api.py

- launch the monitoring_api with

        python run_monitoring_api.py

- launch the retraining_api with

        python run_retraining_api.py

Now all services should be running locally and you can interact with them in the same way as with the dockerized services. 

> NOTE: You may have to run the [`populate_timescale.py`](ml_data_loader/populate_timescale.py) script again in case you had removed the databases before (e.g. via docker-compose down). If databases had not been removed, note that the service and model information persists in the MongoDB, which is why the Django sqliteDB is now out-of-sync (no model/service information persisted). To resolve this issue, simply upload and create the model and the service again via Django admin as described above.

# Useful Links
- https://fastapi.tiangolo.com/
- https://aio-pika.readthedocs.io/en/latest/rabbitmq-tutorial/2-work-queues.html
- https://stackoverflow.com/questions/37512182/how-can-i-periodically-execute-a-function-with-asyncio
- https://tjtelan.com/blog/how-to-link-multiple-docker-compose-via-network/
- https://thinkinfi.com/integrate-plotly-dash-in-django/
- https://medium.com/@emlynoregan/serialising-all-the-functions-in-python-cd880a63b591
- https://mccarthysean.dev/001-05-timescale
- https://pycaret.gitbook.io/docs/get-started/quickstart#regression
- https://docs.evidentlyai.com/get-started/tutorial
- https://docs.evidentlyai.com/user-guide/input-data/data-requirements
