# wine_me_up
1) create the env file and store the environment variables
2) create the core/config.py which will store all the settings of the applications
3) create the core/security.py to create and validate the token, used for authentication
4) create core/dependencies.py to inject logic for API key and JWT token validation
5) create core/exceptions.py for custom exception handler
6) create api/routes_auth.py to define the login route for user auth using JWT
7) work on the data and experiment using jupyter notebooks
8) create training/utils.py for writing convinience functions
9) create training/perform_eda.py to store eda results
10) create training/perform_model_analysis.py to store model comparison results
11) create training/train_model.py for creating models that will be used to predict
12) create the app/cache/redis_cache.py for creating the cache check and store 
13) create app/services/model_service.py to predict value using model
14) create app/api/routes_predict.py to connect with router
15) create middleware/logging_middleware.py for creating the logging
16) create main.py
17) create frontend.py
18) test run locally then deploy on your server
19) create dockerfile for api and frontend
20) create docker-compose.yml
21) create requirements.txt