import logging
import re
import ast
from pathlib import Path
from typing import Any, Dict, List, Union
import json
from datetime import datetime
import markdown
from bs4 import BeautifulSoup

LOGGER = logging.getLogger("aks")

class DocumentationGenerator:
    def __init__(self, repo_path: Path, knowledge_processor: Any):
        self.repo_path = repo_path.resolve()
        self.knowledge_processor = knowledge_processor
        self.docs_dir = self.repo_path / "docs"
        self.templates = self._load_templates()
        self._setup_directories()
        
    def _setup_directories(self) -> None:
        try:
            self.docs_dir.mkdir(exist_ok=True)
            (self.docs_dir / "versions").mkdir(exist_ok=True)
            (self.docs_dir / "assets").mkdir(exist_ok=True)
            LOGGER.info("Documentation directories initialized")
        except Exception as e:
            LOGGER.error(f"Failed to create documentation directories: {e}")
            raise RuntimeError("Documentation setup failed") from e
            
    def _load_templates(self) -> Dict[str, str]:
        return {
            "module": """
## {name}

**Location**: `{path}`  
**Last Updated**: {last_updated}  
**Description**: {description}

### Functions
{functions}

### Classes
{classes}

### Usage Examples
{examples}
            """,
            "function": """
#### {name}

```python
{signature}
```

{description}
            """,
            "class": """
### {name}

```python
{class_signature}
```

{description}

#### Methods
{methods}
            """,
            "method": """
#### {name}

```python
{signature}
```

{description}
            """
        }
    
    def parse_python_file(self, file_path: Path) -> Dict[str, Any]:
        result = {
            "path": str(file_path.relative_to(self.repo_path)),
            "last_updated": datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            "description": "",
            "functions": [],
            "classes": [],
            "examples": []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            module = ast.parse(content)
            result["description"] = ast.get_docstring(module) or ""
            
            for node in module.body:
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        "name": node.name,
                        "signature": self._get_function_signature(node),
                        "description": ast.get_docstring(node) or ""
                    }
                    result["functions"].append(func_info)
                    
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "class_signature": f"class {node.name}",
                        "description": ast.get_docstring(node) or "",
                        "methods": []
                    }
                    
                    for subnode in node.body:
                        if isinstance(subnode, ast.FunctionDef):
                            method_info = {
                                "name": subnode.name,
                                "signature": self._get_function_signature(subnode),
                                "description": ast.get_docstring(subnode) or ""
                            }
                            class_info["methods"].append(method_info)
                    
                    result["classes"].append(class_info)
                    
        except Exception as e:
            LOGGER.error(f"Error parsing {file_path}: {str(e)}")
            result["error"] = str(e)
            
        return result
    
    def _get_function_signature(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        args = [arg.arg for arg in node.args.args]
        defaults = [ast.unparse(d) for d in node.args.defaults] if node.args.defaults else []
        
        positional_args = []
        for i, arg in enumerate(args):
            if i >= len(args) - len(defaults):
                default_index = i - (len(args) - len(defaults))
                positional_args.append(f"{arg}={defaults[default_index]}")
            else:
                positional_args.append(arg)
        
        kwonlyargs = [arg.arg for arg in node.args.kwonlyargs]
        kw_defaults = [ast.unparse(d) if d else None for d in node.args.kw_defaults]
        kw_args = []
        for i, arg in enumerate(kwonlyargs):
            if kw_defaults[i] is not None:
                kw_args.append(f"{arg}={kw_defaults[i]}")
            else:
                kw_args.append(arg)
                
        all_args = positional_args
        if node.args.vararg:
            all_args.append(f"*{node.args.vararg.arg}")
        elif kw_args:
            all_args.append("*")
            
        all_args.extend(kw_args)
        
        if node.args.kwarg:
            all_args.append(f"**{node.args.kwarg.arg}")
            
        async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        return f"{async_prefix}def {node.name}({', '.join(all_args)}):"
    
    def generate_module_documentation(self, module_info: Dict[str, Any]) -> str:
        functions_str = "\n".join([
            self.templates["function"].format(
                name=func["name"],
                signature=func["signature"],
                description=func["description"]
            )
            for func in module_info["functions"]
        ]) or "No functions"
        
        classes_str = "\n".join([
            self.templates["class"].format(
                name=cls["name"],
                class_signature=cls["class_signature"],
                description=cls["description"],
                methods="\n".join([
                    self.templates["method"].format(
                        name=method["name"],
                        signature=method["signature"],
                        description=method["description"]
                    )
                    for method in cls["methods"]
                ]) or "No methods"
            )
            for cls in module_info["classes"]
        ]) or "No classes"
        
        examples_str = "\n".join([
            f"```python\n{example}\n```"
            for example in module_info.get("examples", [])
        ]) or "No examples available"
        
        return self.templates["module"].format(
            name=Path(module_info["path"]).name,
            path=module_info["path"],
            last_updated=module_info["last_updated"],
            description=module_info["description"],
            functions=functions_str,
            classes=classes_str,
            examples=examples_str
        )
    
    def generate_html(self, markdown_content: str) -> str:
        html_content = markdown.markdown(markdown_content)
        
        soup = BeautifulSoup(html_content, 'html.parser')
        toc = []
        headers = soup.find_all(['h2', 'h3', 'h4'])
        
        for header in headers:
            if header.get('id') is None:
                header_id = re.sub(r'\W+', '-', header.text.lower())
                header['id'] = header_id
                
            toc.append(f'<li><a href="#{header_id}">{header.text}</a></li>')
        
        toc_html = f"<div class='toc'><ul>{''.join(toc)}</ul></div>" if toc else ""
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AKS Documentation</title>
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .toc {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        code {{ background: #f0f0f0; padding: 2px 5px; border-radius: 3px; }}
        pre {{ background: #f8f8f8; padding: 10px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>Autonomous Knowledge System Documentation</h1>
    {toc_html}
    {html_content}
</body>
</html>"""
    
    def generate_documentation(self, output_formats: List[str] = ['md', 'html', 'json']) -> None:
        LOGGER.info("Starting documentation generation")
        
        py_files = list(self.repo_path.rglob('*.py'))
        if not py_files:
            LOGGER.warning("No Python files found in repository")
            return
        
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_dir = self.docs_dir / "versions" / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        latest_dir = self.docs_dir / "latest"
        if latest_dir.exists():
            latest_dir.unlink()
        latest_dir.symlink_to(version_dir, target_is_directory=True)
        
        all_docs = {}
        for py_file in py_files:
            if any(part.startswith('.') or part == '__pycache__' for part in py_file.parts):
                continue
                
            LOGGER.info(f"Processing file: {py_file.relative_to(self.repo_path)}")
            module_info = self.parse_python_file(py_file)
            all_docs[str(py_file.relative_to(self.repo_path))] = module_info
            
            md_content = self.generate_module_documentation(module_info)
            base_name = py_file.relative_to(self.repo_path).with_suffix('')
            
            if 'md' in output_formats:
                md_path = version_dir / f"{base_name}.md"
                md_path.parent.mkdir(parents=True, exist_ok=True)
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(md_content)
            
            if 'html' in output_formats:
                html_content = self.generate_html(md_content)
                html_path = version_dir / f"{base_name}.html"
                html_path.parent.mkdir(parents=True, exist_ok=True)
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            if 'json' in output_formats:
                json_path = version_dir / f"{base_name}.json"
                json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(module_info, f, indent=2)
        
        self._generate_index(all_docs, version_dir, output_formats)
        LOGGER.info(f"Documentation generated successfully in {version_dir}")
    
    def _generate_index(self, all_docs: Dict[str, Any], version_dir: Path, output_formats: List[str]) -> None:
        index_content = "# AKS Documentation Index\n\n"
        index_content += f"**Generated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        index_content += "## Modules\n\n"
        
        if self.knowledge_processor:
            try:
                kb_items = self.knowledge_processor.get_recent_items(limit=5)
                index_content += "## Knowledge Base References\n\n"
                for item in kb_items:
                    index_content += f"- [{item.get('title', 'Untitled')}]({item.get('url', '#')})\n"
                index_content += "\n"
            except Exception as e:
                LOGGER.error(f"Failed to integrate knowledge base: {str(e)}")
        
        for file_path, module_info in all_docs.items():
            base_name = Path(file_path).with_suffix('')
            index_content += f"### `{file_path}`\n"
            index_content += f"- **Last Updated**: {module_info['last_updated']}\n"
            index_content += f"- [Markdown]({base_name}.md) | "
            index_content += f"[HTML]({base_name}.html) | "
            index_content += f"[JSON]({base_name}.json)\n"
            index_content += f"{module_info['description'][:200]}...\n\n"
        
        if 'md' in output_formats:
            with open(version_dir / "index.md", 'w', encoding='utf-8') as f:
                f.write(index_content)
        
        if 'html' in output_formats:
            html_content = self.generate_html(index_content)
            with open(version_dir / "index.html", 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        if 'json' in output_formats:
            index_data = {
                "generated_at": datetime.now().isoformat(),
                "modules": list(all_docs.keys())
            }
            with open(version_dir / "index.json", 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2)
