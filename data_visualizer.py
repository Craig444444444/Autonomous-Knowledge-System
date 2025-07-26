import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import shutil

LOGGER = logging.getLogger("aks")

class DataVisualizer:
    """
    Enhanced data visualization component for the Autonomous Knowledge System (AKS).
    Supports multiple visualization backends and output formats with intelligent
    automatic chart type selection based on data characteristics.
    """
    
    def __init__(self, output_dir: Path = Path("/content/visualizations")):
        """
        Initialize the DataVisualizer with configuration.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir.resolve()
        self._setup_output_dir()
        
        # Visualization configuration
        self.style_config = {
            'matplotlib': 'seaborn',
            'plotly_template': 'plotly_white',
            'default_font': 'Arial',
            'color_palette': 'viridis'
        }
        
        LOGGER.info(f"DataVisualizer initialized with output directory: {self.output_dir}")

    def _setup_output_dir(self):
        """Create and secure the output directory."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir.chmod(0o755)  # Restrictive permissions
        except Exception as e:
            LOGGER.error(f"Failed to create visualization directory: {e}")
            raise RuntimeError("Could not initialize visualization output directory")

    def visualize_data(
        self,
        data: Union[Dict, List, pd.DataFrame],
        visualization_type: Optional[str] = None,
        title: str = "AKS Data Visualization",
        output_format: str = "html",
        **kwargs
    ) -> Optional[Path]:
        """
        Main visualization method with automatic type detection and format support.
        
        Args:
            data: Input data (dict, list, or DataFrame)
            visualization_type: Optional forced type (auto-detected if None)
            title: Visualization title
            output_format: One of ['html', 'png', 'jpg', 'json']
            **kwargs: Visualization-specific parameters
            
        Returns:
            Path to saved visualization file or None if failed
        """
        try:
            # Convert input data to consistent format
            df = self._normalize_data(data)
            if df.empty:
                LOGGER.warning("No valid data provided for visualization")
                return None

            # Auto-detect visualization type if not specified
            if visualization_type is None:
                visualization_type = self._auto_detect_visualization_type(df)
                LOGGER.debug(f"Auto-selected visualization type: {visualization_type}")

            # Generate visualization
            viz_func = {
                'line': self._create_line_visualization,
                'bar': self._create_bar_visualization,
                'histogram': self._create_histogram,
                'scatter': self._create_scatter_plot,
                'heatmap': self._create_heatmap,
                'pie': self._create_pie_chart,
                'box': self._create_box_plot,
                'timeseries': self._create_timeseries_plot
            }.get(visualization_type.lower(), self._create_line_visualization)

            fig = viz_func(df, title=title, **kwargs)
            if fig is None:
                LOGGER.error(f"Failed to create {visualization_type} visualization")
                return None

            # Save visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aks_viz_{visualization_type}_{timestamp}.{output_format}"
            output_path = self.output_dir / filename
            
            return self._save_visualization(fig, output_path, output_format)

        except Exception as e:
            LOGGER.error(f"Visualization failed: {e}", exc_info=True)
            return None

    def _normalize_data(self, data: Union[Dict, List, pd.DataFrame]) -> pd.DataFrame:
        """Convert various input formats to standardized DataFrame."""
        try:
            if isinstance(data, pd.DataFrame):
                return data.copy()
            elif isinstance(data, dict):
                return pd.DataFrame.from_dict(data)
            elif isinstance(data, list):
                return pd.DataFrame(data)
            else:
                LOGGER.warning(f"Unsupported data type: {type(data)}")
                return pd.DataFrame()
        except Exception as e:
            LOGGER.error(f"Data normalization failed: {e}")
            return pd.DataFrame()

    def _auto_detect_visualization_type(self, df: pd.DataFrame) -> str:
        """Intelligently select visualization type based on data characteristics."""
        try:
            # Basic heuristics for visualization selection
            numeric_cols = df.select_dtypes(include=np.number).columns
            time_cols = df.select_dtypes(include=['datetime', 'timedelta']).columns
            cat_cols = df.select_dtypes(exclude=np.number).columns
            
            if len(time_cols) > 0 and len(numeric_cols) > 0:
                return 'timeseries'
            elif len(numeric_cols) >= 2:
                if len(df) > 1000:
                    return 'scatter'
                return 'line'
            elif len(cat_cols) > 0 and len(numeric_cols) > 0:
                if len(cat_cols) == 1 and len(numeric_cols) == 1:
                    return 'bar'
                return 'heatmap'
            elif len(numeric_cols) == 1:
                return 'histogram'
            else:
                return 'bar'
        except Exception as e:
            LOGGER.warning(f"Visualization type detection failed, using line: {e}")
            return 'line'

    def _create_line_visualization(self, df: pd.DataFrame, **kwargs) -> Optional[go.Figure]:
        """Create interactive line chart visualization."""
        try:
            title = kwargs.get('title', 'AKS Line Chart')
            x_col = kwargs.get('x', df.columns[0])
            y_cols = kwargs.get('y', [col for col in df.columns if col != x_col])
            
            if isinstance(y_cols, str):
                y_cols = [y_cols]
                
            fig = go.Figure()
            
            for y_col in y_cols:
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    name=y_col,
                    mode='lines+markers',
                    line=dict(width=2)
                ))
                
            fig.update_layout(
                title=title,
                xaxis_title=x_col,
                yaxis_title="Value",
                template=self.style_config['plotly_template'],
                hovermode="x unified"
            )
            
            return fig
        except Exception as e:
            LOGGER.error(f"Line visualization failed: {e}")
            return None

    def _create_bar_visualization(self, df: pd.DataFrame, **kwargs) -> Optional[go.Figure]:
        """Create interactive bar chart visualization."""
        try:
            title = kwargs.get('title', 'AKS Bar Chart')
            x_col = kwargs.get('x', df.columns[0])
            y_col = kwargs.get('y', df.columns[1])
            orientation = kwargs.get('orientation', 'v')
            
            fig = go.Figure()
            
            if orientation == 'v':
                fig.add_trace(go.Bar(
                    x=df[x_col],
                    y=df[y_col],
                    name=y_col
                ))
                xaxis_title = x_col
                yaxis_title = y_col
            else:
                fig.add_trace(go.Bar(
                    y=df[x_col],
                    x=df[y_col],
                    name=y_col,
                    orientation='h'
                ))
                xaxis_title = y_col
                yaxis_title = x_col
                
            fig.update_layout(
                title=title,
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                template=self.style_config['plotly_template']
            )
            
            return fig
        except Exception as e:
            LOGGER.error(f"Bar visualization failed: {e}")
            return None

    def _create_histogram(self, df: pd.DataFrame, **kwargs) -> Optional[go.Figure]:
        """Create histogram visualization."""
        try:
            title = kwargs.get('title', 'AKS Histogram')
            col = kwargs.get('column', df.select_dtypes(include=np.number).columns[0])
            bins = kwargs.get('bins', min(30, len(df[col].unique())))
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=df[col],
                nbinsx=bins,
                marker_color=self.style_config['color_palette']
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title=col,
                yaxis_title="Count",
                template=self.style_config['plotly_template']
            )
            
            return fig
        except Exception as e:
            LOGGER.error(f"Histogram visualization failed: {e}")
            return None

    def _create_scatter_plot(self, df: pd.DataFrame, **kwargs) -> Optional[go.Figure]:
        """Create scatter plot visualization."""
        try:
            title = kwargs.get('title', 'AKS Scatter Plot')
            x_col = kwargs.get('x', df.columns[0])
            y_col = kwargs.get('y', df.columns[1])
            size_col = kwargs.get('size')
            color_col = kwargs.get('color')
            
            fig = go.Figure()
            
            scatter_args = {
                'x': df[x_col],
                'y': df[y_col],
                'mode': 'markers',
                'marker': {}
            }
            
            if size_col:
                scatter_args['marker']['size'] = df[size_col]
            if color_col:
                scatter_args['marker']['color'] = df[color_col]
                scatter_args['marker']['colorscale'] = self.style_config['color_palette']
                scatter_args['marker']['showscale'] = True
                
            fig.add_trace(go.Scatter(**scatter_args))
            
            fig.update_layout(
                title=title,
                xaxis_title=x_col,
                yaxis_title=y_col,
                template=self.style_config['plotly_template']
            )
            
            return fig
        except Exception as e:
            LOGGER.error(f"Scatter plot visualization failed: {e}")
            return None

    def _create_heatmap(self, df: pd.DataFrame, **kwargs) -> Optional[go.Figure]:
        """Create correlation heatmap visualization."""
        try:
            title = kwargs.get('title', 'AKS Heatmap')
            numeric_df = df.select_dtypes(include=np.number)
            
            if len(numeric_df.columns) < 2:
                LOGGER.warning("Not enough numeric columns for heatmap")
                return self._create_bar_visualization(df, **kwargs)
                
            corr_matrix = numeric_df.corr()
            
            fig = go.Figure()
            
            fig.add_trace(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale=self.style_config['color_palette']
            ))
            
            fig.update_layout(
                title=title,
                template=self.style_config['plotly_template'],
                xaxis_showgrid=False,
                yaxis_showgrid=False
            )
            
            return fig
        except Exception as e:
            LOGGER.error(f"Heatmap visualization failed: {e}")
            return None

    def _create_pie_chart(self, df: pd.DataFrame, **kwargs) -> Optional[go.Figure]:
        """Create pie chart visualization."""
        try:
            title = kwargs.get('title', 'AKS Pie Chart')
            values_col = kwargs.get('values', df.select_dtypes(include=np.number).columns[0])
            names_col = kwargs.get('names', df.select_dtypes(exclude=np.number).columns[0])
            
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=df[names_col],
                values=df[values_col],
                textinfo='percent+label'
            ))
            
            fig.update_layout(
                title=title,
                template=self.style_config['plotly_template']
            )
            
            return fig
        except Exception as e:
            LOGGER.error(f"Pie chart visualization failed: {e}")
            return None

    def _create_box_plot(self, df: pd.DataFrame, **kwargs) -> Optional[go.Figure]:
        """Create box plot visualization."""
        try:
            title = kwargs.get('title', 'AKS Box Plot')
            y_cols = kwargs.get('y', df.select_dtypes(include=np.number).columns)
            
            if isinstance(y_cols, str):
                y_cols = [y_cols]
                
            fig = go.Figure()
            
            for col in y_cols:
                fig.add_trace(go.Box(
                    y=df[col],
                    name=col,
                    boxpoints='outliers'
                ))
                
            fig.update_layout(
                title=title,
                template=self.style_config['plotly_template'],
                yaxis_title="Value"
            )
            
            return fig
        except Exception as e:
            LOGGER.error(f"Box plot visualization failed: {e}")
            return None

    def _create_timeseries_plot(self, df: pd.DataFrame, **kwargs) -> Optional[go.Figure]:
        """Create time series visualization."""
        try:
            title = kwargs.get('title', 'AKS Time Series')
            time_col = kwargs.get('time', df.select_dtypes(include=['datetime']).columns[0])
            y_cols = kwargs.get('y', [col for col in df.columns if col != time_col])
            
            if isinstance(y_cols, str):
                y_cols = [y_cols]
                
            fig = go.Figure()
            
            for col in y_cols:
                fig.add_trace(go.Scatter(
                    x=df[time_col],
                    y=df[col],
                    name=col,
                    mode='lines+markers'
                ))
                
            fig.update_layout(
                title=title,
                xaxis_title=time_col,
                template=self.style_config['plotly_template'],
                hovermode="x unified"
            )
            
            return fig
        except Exception as e:
            LOGGER.error(f"Time series visualization failed: {e}")
            return None

    def _save_visualization(self, fig: go.Figure, output_path: Path, output_format: str) -> Optional[Path]:
        """Save visualization to file in specified format."""
        try:
            if output_format == 'html':
                fig.write_html(output_path)
            elif output_format == 'png':
                fig.write_image(output_path, engine="kaleido")
            elif output_format == 'jpg':
                fig.write_image(output_path, format='jpeg', engine="kaleido")
            elif output_format == 'json':
                fig.write_json(output_path)
            else:
                LOGGER.warning(f"Unsupported output format: {output_format}, defaulting to HTML")
                output_path = output_path.with_suffix('.html')
                fig.write_html(output_path)
                
            LOGGER.info(f"Visualization saved to: {output_path}")
            return output_path
        except Exception as e:
            LOGGER.error(f"Failed to save visualization: {e}")
            return None

    def generate_dashboard(self, visualizations: List[Dict], dashboard_title: str = "AKS Dashboard") -> Optional[Path]:
        """
        Generate an interactive HTML dashboard containing multiple visualizations.
        
        Args:
            visualizations: List of visualization configurations
            dashboard_title: Title for the dashboard
            
        Returns:
            Path to saved dashboard HTML file or None if failed
        """
        try:
            # Create temporary directory for components
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Generate all visualizations
                viz_files = []
                for viz_config in visualizations:
                    viz_path = self.visualize_data(
                        **viz_config,
                        output_format='html',
                        output_dir=temp_path
                    )
                    if viz_path:
                        viz_files.append(viz_path)
                
                if not viz_files:
                    LOGGER.warning("No valid visualizations generated for dashboard")
                    return None
                
                # Create dashboard HTML
                dashboard_path = self.output_dir / f"aks_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                
                with open(dashboard_path, 'w') as f:
                    f.write(f"<!DOCTYPE html>\n<html>\n<head>\n<title>{dashboard_title}</title>\n")
                    f.write('<meta charset="UTF-8">\n')
                    f.write('<meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
                    f.write('<style>\n.viz-container {{ width: 100%; margin: 20px 0; }}\n</style>\n</head>\n<body>\n')
                    f.write(f'<h1 style="text-align: center;">{dashboard_title}</h1>\n')
                    
                    for viz_file in viz_files:
                        with open(viz_file, 'r') as viz_f:
                            viz_html = viz_f.read()
                            # Extract just the div and script parts
                            div_start = viz_html.find('<div id="')
                            script_start = viz_html.find('<script type="text/javascript">')
                            script_end = viz_html.rfind('</script>') + 9
                            
                            if div_start != -1 and script_start != -1:
                                viz_content = viz_html[div_start:script_end]
                                f.write(f'<div class="viz-container">{viz_content}</div>\n')
                    
                    f.write('</body>\n</html>')
                
                LOGGER.info(f"Dashboard saved to: {dashboard_path}")
                return dashboard_path
                
        except Exception as e:
            LOGGER.error(f"Dashboard generation failed: {e}", exc_info=True)
            return None

    def cleanup_visualizations(self, max_age_days: int = 30) -> int:
        """
        Clean up old visualization files.
        
        Args:
            max_age_days: Maximum age in days to keep files
            
        Returns:
            Number of files removed
        """
        try:
            cutoff_time = datetime.now().timestamp() - (max_age_days * 86400)
            removed_count = 0
            
            for viz_file in self.output_dir.glob('*'):
                if viz_file.is_file() and viz_file.stat().st_mtime < cutoff_time:
                    try:
                        viz_file.unlink()
                        removed_count += 1
                    except Exception as e:
                        LOGGER.warning(f"Failed to remove {viz_file.name}: {e}")
            
            LOGGER.info(f"Removed {removed_count} old visualization files")
            return removed_count
        except Exception as e:
            LOGGER.error(f"Visualization cleanup failed: {e}")
            return 0
