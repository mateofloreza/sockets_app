## Create the http.server

```
cd static/
python3 -m http.server 8000
```

Then run the server.py

```
cd ..
python server.py
```

Open the web on http://192.168.178.25:8000/client.html

Create the directory to store badge image:

```
mkdir images
twistd3 web --listen tcp:9595 --path .
```

To deploy the LLM, please follow: https://github.com/baycarbone/badge_name_recognition/tree/main/multimodal_model

Run: 

```
python server.py
```