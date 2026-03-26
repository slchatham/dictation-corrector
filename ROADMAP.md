# Dictée Correcteur v2 — Roadmap

## En cours / Fait
- [x] Pipeline Micro → Parakeet TDT 0.6b → Qwen3.5:4b → fenêtre flottante
- [x] Mode STREAMING (chunks 2s) + mode FICHIER (enregistrement batch)
- [x] IPC bidirectionnel daemon ↔ UI (JSON-lines via stdin/stdout)
- [x] Bouton Mute/Unmute
- [x] Vérification de l'environnement au démarrage (preflight)
- [x] Logs horodatés avec niveaux
- [x] Suppression des warnings NeMo/Lightning
- [x] Prompt bilingue — préserve l'anglais intentionnel, ne traduit pas

---

## À faire

### UX / Interface
- [x] **Interface en anglais** — tous les labels, boutons, statuts, commentaires et nom du script en anglais
- [x] **Bouton Exit** — arrêt propre de l'application depuis la fenêtre (évite Ctrl-C)
- [x] **Palette color-blind friendly** — orange (#fab387) remplace rouge, bleu (#89b4fa) remplace vert ; icônes 🎙/🔇 ⏺/⏹ comme signal de forme indépendant

### Audio / Transcription
- [x] **Augmenter CHUNK_SECONDS** (4s) — donne plus de contexte à Parakeet, réduit les coupures au milieu des phrases et améliore la ponctuation automatique
- [x] **Bouton Effacer** — vide le buffer de transcription sans redémarrer le daemon
- [ ] **Import fichier audio** — bouton pour ouvrir un WAV/MP3 (file picker), transcrire via Parakeet, puis corriger via Qwen ; résultat affiché dans la fenêtre comme les autres modes

### Projet
- [x] **git + .gitignore** — initialiser le dépôt, ignorer `__pycache__`, `.nemo`, modèles cachés, `.env`
- [ ] **README** — installation, dépendances, lancement, architecture, raccourcis clavier
