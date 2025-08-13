"""
Script pour télécharger et préparer le dataset bank-assistant-qa
"""

import os
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from config import data_config
import json

def download_and_analyze_dataset():
    """Télécharge le dataset et analyse sa structure"""
    print(f"Téléchargement du dataset {data_config.dataset_name}...")
    
    try:
        # Télécharger le dataset
        dataset = load_dataset(data_config.dataset_name)
        
        print("Dataset téléchargé avec succès!")
        print(f"Structure du dataset: {dataset}")
        
        # Analyser la structure
        if 'train' in dataset:
            train_data = dataset['train']
            print(f"\nNombre d'exemples dans train: {len(train_data)}")
            print(f"Colonnes disponibles: {train_data.column_names}")
            
            # Afficher quelques exemples
            print("\nPremiers exemples:")
            for i in range(min(3, len(train_data))):
                print(f"\nExemple {i+1}:")
                for col in train_data.column_names:
                    print(f"  {col}: {train_data[i][col]}")
        
        return dataset
        
    except Exception as e:
        print(f"Erreur lors du téléchargement: {e}")
        return None

def remove_instruction_column(dataset):
    """Supprime la colonne 'instruction' comme demandé"""
    print("\nSuppression de la colonne 'instruction'...")
    
    processed_dataset = {}
    
    for split_name, split_data in dataset.items():
        if 'instruction' in split_data.column_names:
            # Supprimer la colonne instruction
            processed_split = split_data.remove_columns(['instruction'])
            print(f"Colonne 'instruction' supprimée du split '{split_name}'")
        else:
            processed_split = split_data
            print(f"Pas de colonne 'instruction' trouvée dans le split '{split_name}'")
        
        processed_dataset[split_name] = processed_split
        print(f"Colonnes restantes dans '{split_name}': {processed_split.column_names}")
    
    return DatasetDict(processed_dataset)

def create_training_format(dataset):
    """Formate les données pour l'entraînement"""
    print("\nFormatage des données pour l'entraînement...")
    
    def format_example(example):
        # Créer un prompt structuré pour le chatbot bancaire
        if 'question' in example and 'answer' in example:
            prompt = f"Question: {example['question']}\nRéponse: {example['answer']}"
        elif 'input' in example and 'output' in example:
            prompt = f"Question: {example['input']}\nRéponse: {example['output']}"
        else:
            # Essayer de détecter automatiquement les colonnes
            cols = list(example.keys())
            if len(cols) >= 2:
                prompt = f"Question: {example[cols[0]]}\nRéponse: {example[cols[1]]}"
            else:
                prompt = str(example)
        
        return {"text": prompt}
    
    formatted_dataset = {}
    for split_name, split_data in dataset.items():
        formatted_split = split_data.map(format_example)
        formatted_dataset[split_name] = formatted_split
        print(f"Split '{split_name}' formaté: {len(formatted_split)} exemples")
    
    return DatasetDict(formatted_dataset)

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Divise le dataset en train/validation/test"""
    print(f"\nDivision du dataset (train: {train_ratio}, val: {val_ratio}, test: {test_ratio})...")
    
    # Si le dataset a déjà des splits, les utiliser
    if len(dataset) > 1:
        print("Dataset déjà divisé, utilisation des splits existants")
        return dataset
    
    # Sinon, créer les splits
    if 'train' in dataset:
        data = dataset['train']
    else:
        # Prendre le premier split disponible
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
    
    # Convertir en DataFrame pour faciliter la division
    df = data.to_pandas()
    
    # Première division: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_ratio + test_ratio), 
        random_state=42
    )
    
    # Deuxième division: val vs test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=test_ratio/(val_ratio + test_ratio), 
        random_state=42
    )
    
    # Convertir en Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    result = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    print(f"Train: {len(train_dataset)} exemples")
    print(f"Validation: {len(val_dataset)} exemples")
    print(f"Test: {len(test_dataset)} exemples")
    
    return result

def save_processed_dataset(dataset, output_dir="./processed_data"):
    """Sauvegarde le dataset traité"""
    print(f"\nSauvegarde du dataset dans {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder chaque split
    for split_name, split_data in dataset.items():
        split_path = os.path.join(output_dir, f"{split_name}.json")
        split_data.to_json(split_path)
        print(f"Split '{split_name}' sauvegardé: {split_path}")
    
    # Sauvegarder les métadonnées
    metadata = {
        "dataset_name": data_config.dataset_name,
        "splits": {name: len(data) for name, data in dataset.items()},
        "columns": dataset['train'].column_names if 'train' in dataset else list(dataset.values())[0].column_names,
        "total_examples": sum(len(data) for data in dataset.values())
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Métadonnées sauvegardées: {metadata_path}")
    return output_dir

def main():
    """Fonction principale"""
    print("=== Préparation du dataset bank-assistant-qa ===\n")
    
    # 1. Télécharger et analyser
    dataset = download_and_analyze_dataset()
    if dataset is None:
        return
    
    # 2. Supprimer la colonne instruction
    dataset = remove_instruction_column(dataset)
    
    # 3. Formater pour l'entraînement
    dataset = create_training_format(dataset)
    
    # 4. Diviser en train/val/test
    dataset = split_dataset(dataset)
    
    # 5. Sauvegarder
    output_dir = save_processed_dataset(dataset)
    
    print(f"\n=== Préparation terminée ===")
    print(f"Dataset traité disponible dans: {output_dir}")
    print(f"Prêt pour le fine-tuning LoRA!")

if __name__ == "__main__":
    main()
