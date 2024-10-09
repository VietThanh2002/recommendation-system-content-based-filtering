Create new virtual environment

```bash
python –m venv .venv
```
Activate the virtual environment (Using PowerShell)

```bash
.venv\Scripts\Activate.ps1
```
Install dependences

```bash
pip install -r .\requirements.txt
```
Run model

```bash
py content_based_recommendations.py 
```

Call to system
```bash
    http://127.0.0.1:9090/recommendations?product_id = id_product
```


