"""
Téléchargement simple et direct de Llama 3.1 8B
"""

import os
from pathlib import Path

def main():
    print("🦙 Téléchargement Simple Llama 3.1 8B")
    print("=" * 40)
    
    # Configuration
    token = "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"
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    local_dir = "./models/Llama-3.1-8B-Instruct"
    
    # Créer le répertoire
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Modèle: {model_id}")
    print(f"📁 Destination: {local_dir}")
    
    try:
        from huggingface_hub import snapshot_download
        
        print("\n🚀 Début du téléchargement...")
        
        # Téléchargement simple
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            token=token
        )
        
        print("\n✅ Téléchargement terminé!")
        
        # Vérifier les fichiers
        files = list(Path(local_dir).glob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file()) / (1024**3)
        
        print(f"📊 Résultat:")
        print(f"  📄 Fichiers: {len(files)}")
        print(f"  💾 Taille: {total_size:.1f} GB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    main()
