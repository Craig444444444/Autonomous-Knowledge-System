

### **📝 How to Create Python Files in Colab**

#### **1. What This Does**
Creates `.py` files directly in your Colab runtime to organize your AKS code.

#### **2. How It Works**
Use Colab's `%%writefile` magic command at the top of a cell:

```python
%%writefile filename.py
# Your Python code goes here
def hello():
    print("AKS is running!")
```

#### **3. Step-by-Step Example**
1. **Create a new cell**
2. **Write this code** (example for `aks_main.py`):
```python
%%writefile aks_main.py
import os
from git_manager import GitManager

def main():
    print("Starting AKS...")
    gm = GitManager(os.getenv("GITHUB_TOKEN"))
    gm.initialize_repo()

if __name__ == "__main__":
    main()
```
3. **Run the cell** - This creates the file instantly

#### **4. Verify the File Exists**
```python
!ls -l  # Should show your new file
!cat aks_main.py  # View contents
```

#### **5. Key Notes**
- Files disappear when Colab restarts (use Google Drive for persistence)
- Organize files with folders:
```python
%%writefile core/git_manager.py
# File content here
```

#### **6. Need to Edit?**
Overwrite by running `%%writefile` again with changes.

---

**That's it!** This is all you need to create/modify Python files in Colab. For AKS, you'd repeat this for each module file. 
