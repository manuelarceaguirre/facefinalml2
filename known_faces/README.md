# Known Faces Enrollment Folder

Optional demo-only face recognition layer.

Add consented/public classroom images in subfolders by person name:

```text
known_faces/
  Manuel Arce/
    1.jpg
    2.jpg
  Jane Doe/
    profile.jpg
```

Then run:

```bash
python scripts/enroll_known_faces.py
```

This creates:

```text
models/known_faces.joblib
```

The raw photos in this folder are ignored by git. The enrollment script stores only ArcFace embeddings in the model file. Do not use this without consent.
