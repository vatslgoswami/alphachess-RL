KRK, 50,015 positions, γ = 0.9. Backups for retrograde analysis count predecessor edges walked.

| method | starts | trials_or_sweeps | backups | positions_touched | exact_entries | wrong_entries | seconds | converged |
|---|---|---|---|---|---|---|---|---|
| retrograde analysis | all | 33 levels | 524,507 | 50,015 | 50,015 | 0 | 0.11 | True |
| synchronous value iteration | all | 33 sweeps | 1,650,495 | 50,015 | 50,015 | 0 | 0.15 | True |
| RTDP, single value, V0 = 0 (primer) | 1,000 | 48,322 trials | 2,294,921 | 49,619 | 3,649 | 45,970 | 81.39 | False |
| RTDP with bounds (U0 = +1, L0 = -1) | 10 | 2,371 trials | 605,122 | 49,928 | 11,602 | 38,326 | 25.0 | True |
| RTDP with bounds (U0 = +1, L0 = -1) | 100 | 3,517 trials | 646,850 | 49,923 | 20,403 | 29,520 | 69.4 | True |
| RTDP with bounds (U0 = +1, L0 = -1) | 1,000 | 5,566 trials | 679,356 | 49,923 | 23,802 | 26,121 | 73.48 | True |
