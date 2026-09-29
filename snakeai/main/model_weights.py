from pathlib import Path
import hashlib,json,os

def ensure_model(name):
    path=Path(__file__).resolve().parent/'trained_models_cnn'/name
    folder=path.with_name(path.name+'.parts')
    manifest=json.loads((folder/'manifest.json').read_text())
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sha256']:return path
    print('Preparing model: '+name,flush=True)
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.partial')
    digest=hashlib.sha256()
    try:
        with temporary.open('wb') as output:
            for part in manifest['parts']:
                data=(folder/part['name']).read_bytes()
                if hashlib.sha256(data).hexdigest()!=part['sha256']:raise RuntimeError('Corrupt model part: '+part['name'])
                output.write(data);digest.update(data)
        if digest.hexdigest()!=manifest['sha256']:raise RuntimeError('Model checksum mismatch')
        temporary.replace(path)
    finally:
        if temporary.exists():temporary.unlink()
    return path
