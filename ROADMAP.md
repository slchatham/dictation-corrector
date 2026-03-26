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
- [ ] **Interface en anglais** — tous les labels, boutons, statuts et messages en anglais
- [ ] **Bouton Exit** — arrêt propre de l'application depuis la fenêtre (évite Ctrl-C)
- [ ] **Palette color-blind friendly** — remplacer rouge/vert par des combinaisons accessibles (ex : bleu/orange, avec icônes en complément de la couleur)

### Audio / Transcription
- [ ] **Augmenter CHUNK_SECONDS** (ex : 4-5s) — donne plus de contexte à Parakeet, réduit les coupures au milieu des phrases et améliore la ponctuation automatique
- [ ] **Bouton Effacer** — vide le buffer de transcription sans redémarrer le daemon

### Projet
- [ ] **git + .gitignore** — initialiser le dépôt, ignorer `__pycache__`, `.nemo`, modèles cachés, `.env`
- [ ] **README** — installation, dépendances, lancement, architecture, raccourcis clavier
