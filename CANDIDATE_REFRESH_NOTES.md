# Candidate Status Refresh — May 22, 2026

This package corrects legacy candidate metadata in `inputs/race_inputs.csv`.

Key corrections:
- Minnesota is now treated as an open Democratic-held seat. Tina Smith is retiring and is no longer listed as the Democratic candidate.
- Georgia lists Jon Ossoff as the Democratic incumbent and the GOP field as the Collins/Dooley runoff.
- Texas lists James Talarico as the Democratic nominee and Cornyn/Paxton as the GOP runoff.
- Ohio lists Sherrod Brown vs. Jon Husted.
- Maine lists Susan Collins on the GOP side and Graham Platner / David Costello on the Democratic side.
- Florida remains a special election with Ashley Moody as appointed GOP incumbent and a still-contested Democratic field.

Modeling note:
These candidate labels do not automatically change the forecast margins unless you edit the polling/fundamentals columns. The Python engine treats candidate/status metadata separately from numerical model inputs.