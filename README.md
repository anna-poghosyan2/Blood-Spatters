# Blood Spatters
A machine learning project that uses the random forest algorithm to classify the cause of impact blood spatters as either gunshot or blunt force induced.



# Set-up

Instructions are written for the Linux commandline.

Create a virtual environment (I used python 3.9)
```commandline
 python3 -m venv venv
```

Start-up the virtual environment
```commandline
source venv/bin/activate
```

Upgrade pip and install the requirements
```commandline
pip install pip==22.2.2
pip install -r requirements.txt
```

Train the random forest
```commandline
python randomforest.py
```


