# Day 1 Log - July 15, 2026
**Today's goal:** 
**Start Time:* 5 pm
**End time:** 10 pm
---

## 1. Core Technical Questions

### Q1: What is the technical difference between self-attention and multi-head attention?

### 


**notes**
I implemented scaled dot‑product attention manually to ensure I fully understand the mechanism, but left a comment noting that F.scaled_dot_product_attention would be used in production for performance. The shape tests validate that all components preserve tensor dimensions correctly.


## 3. Today's Retrospective
* what went well:

* what I struggle with / questions I still have:
 - we made git commit without gitignore, so we had to use git-filter force 
 option to revert clean git 
 
 ```bash
    git filter-repo --path .venv --invert-paths --force
 ```
 checking git size 
```bash
    du -sh .git # 



