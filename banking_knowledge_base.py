"""
Création de la base de connaissances bancaires pour RAG
"""

import os
import json
from typing import List, Dict
from pathlib import Path

def create_banking_documents():
    """Crée une base de documents bancaires complète"""
    
    banking_docs = {
        "comptes_bancaires": {
            "title": "Guide des Comptes Bancaires",
            "content": """
            TYPES DE COMPTES BANCAIRES

            1. COMPTE COURANT
            - Frais de tenue: 2€/mois pour les revenus < 1500€, 5€/mois sinon
            - Découvert autorisé: jusqu'à 500€ (taux 15% annuel)
            - Carte bancaire incluse
            - Virements illimités
            - Chéquier gratuit (50 chèques/an)

            2. COMPTE ÉPARGNE
            - Livret A: taux 3%, plafond 22,950€, exonéré d'impôts
            - Livret Développement Durable: taux 3%, plafond 12,000€
            - Compte épargne logement: taux 2%, prime d'État possible
            - Pas de frais de tenue de compte

            3. COMPTE JOINT
            - Même conditions que compte courant
            - Responsabilité solidaire des titulaires
            - Carte pour chaque titulaire
            - Procuration automatique

            OUVERTURE DE COMPTE
            Documents requis:
            - Pièce d'identité valide
            - Justificatif de domicile (-3 mois)
            - Justificatif de revenus (3 derniers bulletins de salaire)
            - RIB de l'ancien compte (si changement de banque)
            """
        },
        
        "cartes_bancaires": {
            "title": "Guide des Cartes Bancaires",
            "content": """
            TYPES DE CARTES

            1. CARTE BLEUE CLASSIQUE
            - Cotisation: 45€/an
            - Plafond retrait: 300€/7 jours
            - Plafond paiement: 1,500€/30 jours
            - Assurance voyage incluse
            - Paiement sans contact jusqu'à 50€

            2. CARTE GOLD
            - Cotisation: 130€/an
            - Plafond retrait: 800€/7 jours
            - Plafond paiement: 5,000€/30 jours
            - Assurances étendues (voyage, achat)
            - Conciergerie 24h/24

            3. CARTE PRÉPAYÉE
            - Pas de découvert possible
            - Rechargeable
            - Idéale pour les jeunes
            - Frais: 2€/mois

            ACTIVATION ET SÉCURITÉ
            - Activation: composer le 3179 + code confidentiel
            - Opposition: 0892 705 705 (24h/24)
            - Changement de code: dans tous nos distributeurs
            - Paiement en ligne sécurisé avec 3D Secure
            """
        },
        
        "prets_credits": {
            "title": "Guide des Prêts et Crédits",
            "content": """
            PRÊT IMMOBILIER

            Conditions d'éligibilité:
            - Apport personnel: minimum 10% du montant
            - Taux d'endettement: maximum 35% des revenus
            - CDI ou revenus stables depuis 2 ans
            - Âge maximum: 65 ans à la fin du prêt

            Taux actuels (variables selon profil):
            - 15 ans: 3.2% à 3.8%
            - 20 ans: 3.4% à 4.0%
            - 25 ans: 3.6% à 4.2%

            Documents requis:
            - 3 derniers bulletins de salaire
            - Avis d'imposition N-1 et N-2
            - Relevés de compte (3 mois)
            - Compromis de vente
            - Justificatif d'apport personnel

            CRÉDIT CONSOMMATION
            - Montant: 200€ à 75,000€
            - Durée: 3 mois à 7 ans
            - Taux: 2.9% à 8.9% selon montant et durée
            - Réponse sous 48h
            - Remboursement anticipé possible sans frais

            CRÉDIT RENOUVELABLE
            - Réserve d'argent disponible
            - Taux: 15.9% sur l'utilisé
            - Reconstitution automatique
            - Carte de crédit associée possible
            """
        },
        
        "services_numeriques": {
            "title": "Services Bancaires Numériques",
            "content": """
            BANQUE EN LIGNE

            Accès et sécurité:
            - Connexion: www.mabanque.fr
            - Identifiant: numéro de compte
            - Mot de passe: 8 caractères minimum
            - Double authentification par SMS
            - Session automatiquement fermée après 15 min d'inactivité

            Services disponibles 24h/24:
            - Consultation des comptes et historique
            - Virements (France et international)
            - Commande de chéquier
            - Opposition carte bancaire
            - Prise de rendez-vous en agence
            - Simulation de prêts

            APPLICATION MOBILE
            - Disponible iOS et Android
            - Authentification biométrique
            - Paiement mobile sans contact
            - Géolocalisation des distributeurs
            - Notifications en temps réel
            - Dépôt de chèque par photo

            VIREMENTS
            - Virement interne: gratuit et instantané
            - Virement SEPA: 1€, exécuté sous 24h
            - Virement international: 15€ + frais de change
            - Virement permanent: programmation gratuite
            """
        },
        
        "tarifs_frais": {
            "title": "Tarifs et Frais Bancaires",
            "content": """
            FRAIS DE FONCTIONNEMENT

            Tenue de compte:
            - Compte courant: 2€ à 5€/mois selon revenus
            - Compte épargne: gratuit
            - Compte joint: 3€ à 7€/mois

            Moyens de paiement:
            - Carte bleue: 45€/an
            - Carte gold: 130€/an
            - Chéquier: gratuit (50 chèques/an)
            - Chèques supplémentaires: 0.15€/chèque

            Opérations courantes:
            - Virement SEPA: 1€
            - Prélèvement: gratuit
            - TIP: gratuit
            - Virement international: 15€

            INCIDENTS ET IRRÉGULARITÉS
            - Rejet de prélèvement: 20€
            - Rejet de chèque: 30€
            - Commission d'intervention: 8€ (max 80€/mois)
            - Lettre d'information: 15€
            - Dossier d'irrégularité: 50€

            SERVICES SPÉCIAUX
            - Coffre-fort: 60€ à 200€/an selon taille
            - Assurance moyens de paiement: 25€/an
            - Envoi de relevés par courrier: 2€/mois
            - Duplicata de relevé: 5€
            """
        },
        
        "procedures_reclamations": {
            "title": "Procédures et Réclamations",
            "content": """
            RÉCLAMATIONS

            Première étape - Agence:
            - Contacter votre conseiller directement
            - Horaires: lundi-vendredi 9h-17h, samedi 9h-12h
            - Téléphone agence: voir sur votre RIB
            - Email: prenom.nom@mabanque.fr

            Deuxième étape - Service clientèle:
            - Téléphone: 3179 (service gratuit + prix appel)
            - Horaires: 7j/7, 8h-20h
            - Email: reclamation@mabanque.fr
            - Courrier: Service Réclamations, BP 12345, 75001 Paris

            Médiation bancaire:
            - Si désaccord persiste après 2 mois
            - Gratuit pour le client
            - Médiateur indépendant
            - Saisine: mediateur@mabanque.fr

            URGENCES 24H/24
            - Opposition carte: 0892 705 705
            - Urgence à l'étranger: +33 1 42 77 11 90
            - Perte/vol de chéquier: 0892 683 208

            CHANGEMENT DE COORDONNÉES
            - En ligne sur votre espace client
            - En agence avec justificatif
            - Par courrier signé
            - Délai de prise en compte: 48h
            """
        },
        
        "assurances_bancaires": {
            "title": "Assurances et Protections",
            "content": """
            ASSURANCE MOYENS DE PAIEMENT

            Couverture:
            - Utilisation frauduleuse carte bancaire
            - Vol d'espèces au distributeur (max 800€)
            - Perte/vol de clés (remplacement serrures)
            - Assistance juridique
            - Cotisation: 25€/an

            ASSURANCE EMPRUNTEUR
            - Obligatoire pour tout prêt immobilier
            - Garanties: décès, invalidité, incapacité
            - Taux: 0.25% à 0.45% du capital emprunté
            - Possibilité de délégation d'assurance
            - Questionnaire médical requis

            ASSURANCE VIE
            - Placement et transmission
            - Fonds euros garantis: 2.5% net
            - Unités de compte: selon marchés
            - Versements libres à partir de 150€
            - Avantages fiscaux après 8 ans

            PROTECTION JURIDIQUE
            - Défense de vos intérêts
            - Assistance téléphonique
            - Prise en charge frais d'avocat
            - Domaines: famille, consommation, travail
            - Cotisation: 45€/an
            """
        }
    }
    
    return banking_docs

def save_documents_to_files():
    """Sauvegarde les documents dans des fichiers séparés"""
    
    # Créer le dossier de documents
    docs_dir = Path("banking_documents")
    docs_dir.mkdir(exist_ok=True)
    
    banking_docs = create_banking_documents()
    
    # Sauvegarder chaque document
    for doc_id, doc_data in banking_docs.items():
        file_path = docs_dir / f"{doc_id}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# {doc_data['title']}\n\n")
            f.write(doc_data['content'])
        
        print(f"✅ Document sauvegardé: {file_path}")
    
    # Créer un index des documents
    index = {
        "documents": [
            {
                "id": doc_id,
                "title": doc_data["title"],
                "file": f"{doc_id}.txt",
                "description": doc_data["content"][:200] + "..."
            }
            for doc_id, doc_data in banking_docs.items()
        ],
        "total_documents": len(banking_docs),
        "created_at": "2025-01-14"
    }
    
    with open(docs_dir / "index.json", 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Index créé: {docs_dir / 'index.json'}")
    return docs_dir

def create_faq_data():
    """Crée une FAQ bancaire détaillée"""
    
    faq_data = [
        {
            "question": "Comment ouvrir un compte bancaire ?",
            "answer": "Pour ouvrir un compte, vous devez fournir une pièce d'identité, un justificatif de domicile de moins de 3 mois, et vos 3 derniers bulletins de salaire. La procédure prend environ 30 minutes en agence.",
            "category": "comptes",
            "keywords": ["ouverture", "compte", "documents", "procédure"]
        },
        {
            "question": "Quels sont les frais de tenue de compte ?",
            "answer": "Les frais de tenue de compte sont de 2€/mois pour les revenus inférieurs à 1500€, et 5€/mois au-delà. Les comptes épargne sont gratuits.",
            "category": "tarifs",
            "keywords": ["frais", "tenue", "compte", "coût"]
        },
        {
            "question": "Comment activer ma carte bancaire ?",
            "answer": "Pour activer votre carte, composez le 3179 depuis un téléphone fixe et suivez les instructions. Vous devrez saisir votre code confidentiel.",
            "category": "cartes",
            "keywords": ["activation", "carte", "3179", "code"]
        },
        {
            "question": "Que faire en cas de perte de carte ?",
            "answer": "En cas de perte ou vol, faites immédiatement opposition en appelant le 0892 705 705 (24h/24). Une nouvelle carte vous sera envoyée sous 5 jours ouvrés.",
            "category": "cartes",
            "keywords": ["perte", "vol", "opposition", "urgence"]
        },
        {
            "question": "Comment faire un virement ?",
            "answer": "Vous pouvez faire un virement via votre espace en ligne, l'application mobile, ou en agence. Les virements SEPA coûtent 1€ et sont exécutés sous 24h.",
            "category": "services",
            "keywords": ["virement", "transfert", "SEPA", "en ligne"]
        },
        {
            "question": "Quelles sont les conditions pour un prêt immobilier ?",
            "answer": "Il faut un apport de minimum 10%, un taux d'endettement inférieur à 35%, des revenus stables depuis 2 ans, et être âgé de moins de 65 ans à la fin du prêt.",
            "category": "credits",
            "keywords": ["prêt", "immobilier", "conditions", "apport"]
        },
        {
            "question": "Comment contester un prélèvement ?",
            "answer": "Vous avez 8 semaines pour contester un prélèvement SEPA. Contactez votre agence ou utilisez votre espace en ligne pour faire la demande de remboursement.",
            "category": "reclamations",
            "keywords": ["contestation", "prélèvement", "remboursement", "SEPA"]
        },
        {
            "question": "Quels sont les horaires d'ouverture ?",
            "answer": "Les agences sont ouvertes du lundi au vendredi de 9h à 17h, et le samedi de 9h à 12h. Le service client est disponible 7j/7 de 8h à 20h au 3179.",
            "category": "services",
            "keywords": ["horaires", "ouverture", "agence", "service client"]
        }
    ]
    
    # Sauvegarder la FAQ
    faq_file = Path("banking_documents") / "faq.json"
    with open(faq_file, 'w', encoding='utf-8') as f:
        json.dump(faq_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ FAQ sauvegardée: {faq_file}")
    return faq_data

def main():
    """Fonction principale"""
    print("🏦 Création de la base de connaissances bancaires")
    print("=" * 50)
    
    # 1. Créer les documents
    docs_dir = save_documents_to_files()
    
    # 2. Créer la FAQ
    faq_data = create_faq_data()
    
    print(f"\n✅ Base de connaissances créée dans: {docs_dir}")
    print(f"📄 {len(create_banking_documents())} documents principaux")
    print(f"❓ {len(faq_data)} questions FAQ")
    
    print(f"\n📁 Structure créée:")
    print(f"  banking_documents/")
    print(f"  ├── comptes_bancaires.txt")
    print(f"  ├── cartes_bancaires.txt")
    print(f"  ├── prets_credits.txt")
    print(f"  ├── services_numeriques.txt")
    print(f"  ├── tarifs_frais.txt")
    print(f"  ├── procedures_reclamations.txt")
    print(f"  ├── assurances_bancaires.txt")
    print(f"  ├── faq.json")
    print(f"  └── index.json")
    
    print(f"\n🎯 Prêt pour l'étape suivante: Configuration des embeddings")

if __name__ == "__main__":
    main()
