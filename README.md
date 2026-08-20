# ⚽ Winamax Betting & ML Analyzer

Application web interactive construite avec **Streamlit** pour analyser les matchs de football, calculer les probabilités à l'aide de modèles de Machine Learning et détecter les opportunités de paris à valeur ajoutée (**Value Bets**).

## 🚀 Fonctionnalités

- **Filtrage des matchs** : Recherche par date, championnat et affiche les cotes 1N2 en temps réel.
- **Modèles de Machine Learning** :
  - Prédiction des probabilités 1N2 avec calcul de l'Espérance de Gain (EV).
  - Prédiction du marché "Les 2 équipes marquent" (BTTS).
- **Récapitulatif des meilleurs paris** : Synthèse automatique des meilleures opportunités (Value Bets) sur l'onglet principal.
- **Analyse détaillée des marchés annexes** :
  - Tableau des buts (+ / -) avec surlignage des plus fortes probabilités.
  - Tableau des buteurs, passeurs et joueurs décisifs.

---

## 🛠️ Installation en local

1. **Cloner le projet :**
   ```bash
   git clone [https://github.com/votre-utilisateur/votre-depot.git](https://github.com/votre-utilisateur/votre-depot.git)
   cd votre-depot
   
   Créer et activer un environnement virtuel :

Bash
python -m venv venv
# Sur Windows :
venv\Scripts\activate
# Sur macOS / Linux :
source venv/bin/activate
Installer les dépendances :

Bash
pip install -r requirements.txt
Lancer l'application Streamlit :

Bash
streamlit run app.py
☁️ Déploiement
Cette application est prête à être déployée sur Streamlit Community Cloud :

Poussez le code sur GitHub.

Connectez-vous sur share.streamlit.io.

Sélectionnez le dépôt et pointez vers app.py.


---

### 💡 Prochaine étape pour déployer sur GitHub :
Dans votre terminal (à la racine de votre projet) :
1. `git init` *(si pas déjà fait)*
2. `git add .`
3. `git commit -m "Initial commit avec Streamlit app"`
4. `git branch -M main`
5. `git remote add origin [https://github.com/votre-pseudo/votre-nom-de-repo.git](https://github.com/votre-pseudo/votre-nom-de-repo.git)`
6. `git push -u origin main`

Ensuite, rendez-vous sur la page web de Streamlit et cliquez sur **Deploy** !