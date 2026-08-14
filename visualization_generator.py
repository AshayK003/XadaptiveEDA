import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class VisualizationGenerator:
    def generate_visualization(self, analysis_type, df, columns, quality_report=None, **kwargs):
        if df is None or df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for visualization",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid columns selected for visualization",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig
        dispatch = {
            'distribution': self._create_distribution_plot,
            'correlation': self._create_correlation_plot,
            'missing_values': self._create_missing_values_plot,
            'categorical': self._create_categorical_plot,
            'outliers': self._create_outliers_plot,
            'time_series': self._create_time_series_plot,
            'clustering': self._create_clustering_plot,
            'feature_importance': self._create_feature_importance_plot,
        }
        handler = dispatch.get(analysis_type)
        if handler:
            fig = handler(df, valid_cols, **kwargs)
            self._annotate_quality_warnings(fig, valid_cols, quality_report)
            return fig
        fig = go.Figure()
        fig.add_annotation(
            text=f"Visualization for {analysis_type} not implemented",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig

    def _annotate_quality_warnings(self, fig, columns, quality_report):
        """Add annotation to figure if any selected columns have quality issues."""
        if quality_report is None or not columns:
            return
        warnings = []
        for col in columns:
            if col in quality_report.sparse_columns:
                warnings.append(f"'{col}' is sparse ({quality_report.null_percentages.get(col, 0):.0f}% missing)")
            if col in quality_report.constant_columns:
                warnings.append(f"'{col}' is constant (single value)")
            if col in quality_report.mixed_type_columns:
                warnings.append(f"'{col}' has mixed data types")
        if warnings:
            fig.add_annotation(
                text="[Quality warning] " + "; ".join(warnings),
                xref="paper", yref="paper", x=0.5, y=-0.15,
                showarrow=False, font=dict(color="orange", size=11),
                bgcolor="rgba(255,255,255,0.8)"
            )
            # Make room for the annotation
            fig.update_layout(margin=dict(b=60))

    def _create_distribution_plot(self, df, columns, **kwargs):
        if not columns:
            return go.Figure()

        if len(columns) == 1:
            col = columns[0]
            fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Distribution of {col}", f"Boxplot of {col}"])
            data = df[col].dropna() 
            if data.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No data available for this column",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
                return fig
            fig.add_trace(go.Histogram(x=data, name=col, histnorm='probability density'), row=1, col=1)
            fig.add_trace(go.Box(y=data, name=col, boxmean='sd'), row=1, col=2)
            fig.update_layout(height=450, showlegend=False)
            return fig

        n = len(columns)
        fig = make_subplots(
            rows=n, cols=2,
            subplot_titles=[t for col in columns for t in (f"Distribution of {col}", f"Boxplot of {col}")]
        )
        for i, col in enumerate(columns):
            data = df[col].dropna()
            if data.empty:
                fig.add_annotation(
                    text=f"No data available for {col}",
                    x=0.5,
                    y=(i + 0.5) / n,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
                continue
            fig.add_trace(go.Histogram(x=data, showlegend=False, histnorm='probability density'), row=i+1, col=1)
            fig.add_trace(go.Box(y=data, showlegend=False, boxmean='sd'), row=i+1, col=2)
        fig.update_layout(height=350 * n, title_text="Distribution Analysis")
        return fig

    def _create_correlation_plot(self, df, columns, **kwargs):
        method = kwargs.get('method', 'pearson')
        num_df = df[columns].select_dtypes(include=['number'])

        if num_df.empty or num_df.shape[1] < 2:
            fig = go.Figure()
            fig.add_annotation(text="Need at least 2 numerical columns", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

        corr = num_df.corr(method=method)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        tri = corr.copy()
        tri[mask] = np.nan

        fig = make_subplots(rows=1, cols=2, subplot_titles=["Triangular Heatmap", "Full Matrix"], horizontal_spacing=0.08)
        colorscale = 'RdBu_r'

        fig.add_trace(go.Heatmap(z=tri, x=corr.columns, y=corr.index, colorscale=colorscale,
                                 zmin=-1, zmax=1, text=corr.round(2), texttemplate="%{text}", showscale=False), row=1, col=1)
        fig.add_trace(go.Heatmap(z=corr, x=corr.columns, y=corr.index, colorscale=colorscale,
                                 zmin=-1, zmax=1, text=corr.round(2), texttemplate="%{text}"), row=1, col=2)

        fig.update_layout(title_text=f"{method.capitalize()} Correlation", height=550)
        return fig

    def _create_missing_values_plot(self, df, columns, **kwargs):
        missing = pd.DataFrame({
            'column': columns,
            'missing_pct': [df[col].isnull().mean() * 100 for col in columns]
        }).sort_values('missing_pct', ascending=False)

        fig = px.bar(missing, x='column', y='missing_pct', text=missing['missing_pct'].round(1).astype(str) + '%',
                     labels={'missing_pct': 'Missing (%)', 'column': ''}, title='Missing Values by Column')
        fig.update_layout(xaxis={'categoryorder': 'total descending'}, height=450)
        return fig

    def _create_categorical_plot(self, df, columns, **kwargs):
        max_categories = kwargs.get('max_categories', 10)
        plot_type = kwargs.get('plot_type', 'bar')

        if not columns:
            fig = go.Figure()
            fig.add_annotation(text="No categorical columns selected", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

        colors = px.colors.qualitative.Pastel

        if len(columns) == 1:
            col = columns[0]
            vc = df[col].value_counts().nlargest(max_categories)
            if vc.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"No data available for {col}",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
                return fig
            if plot_type == 'pie':
                fig = px.pie(values=vc.values, names=vc.index, title=f'Distribution of {col}',
                             color_discrete_sequence=colors)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=450)
            else:
                fig = px.bar(x=vc.index, y=vc.values, labels={'x': col, 'y': 'Count'},
                             title=f'Distribution of {col}', text=vc.values, color_discrete_sequence=colors)
                fig.update_layout(xaxis={'categoryorder': 'total descending'}, height=450)
            return fig

        if plot_type == 'pie':
            fig = make_subplots(rows=1, cols=len(columns),
                                subplot_titles=[f"Distribution of {col}" for col in columns],
                                specs=[[{"type": "domain"}] * len(columns)])
            for i, col in enumerate(columns):
                vc = df[col].value_counts().nlargest(max_categories)
                c = colors[i % len(colors)]
                fig.add_trace(go.Pie(values=vc.values, labels=vc.index, name=col,
                                     textposition='inside', textinfo='percent+label',
                                     marker_colors=[c] * len(vc)), row=1, col=i+1)
            fig.update_layout(height=400, showlegend=False)
        else:
            fig = make_subplots(rows=len(columns), cols=1,
                                subplot_titles=[f"Distribution of {col}" for col in columns])
            for i, col in enumerate(columns):
                vc = df[col].value_counts().nlargest(max_categories)
                c = colors[i % len(colors)]
                fig.add_trace(go.Bar(x=vc.index, y=vc.values, name=col, text=vc.values,
                                     textposition='auto', marker_color=c), row=i+1, col=1)
            fig.update_layout(height=400 * len(columns), showlegend=False)
        return fig

    def _create_outliers_plot(self, df, columns, **kwargs):
        if not columns:
            fig = go.Figure()
            fig.add_annotation(text="No columns selected", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

        if len(columns) == 1:
            col = columns[0]
            data = df[col].dropna()

            if data.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"No data available for {col}",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
                return fig
            fig = go.Figure()
            fig.add_trace(go.Box(y=data, name=col, boxmean='sd'))
            fig.update_layout(title="Outlier Analysis", height=450)
            return fig

        fig = make_subplots(rows=1, cols=len(columns), subplot_titles=[f"Outliers in {col}" for col in columns])
        for i, col in enumerate(columns):
            data = df[col].dropna()

            if data.empty:
                fig.add_annotation(
                    text=f"No data available for {col}",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
                continue
            fig.add_trace(go.Box(y=data, showlegend=False, boxmean='sd'), row=1, col=i+1)
        fig.update_layout(title_text="Outlier Analysis", height=450)
        return fig

    def _create_time_series_plot(self, df, columns, **kwargs):
        if not columns:
            fig = go.Figure()
            fig.add_annotation(text="No time series columns selected", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

        fig = go.Figure()
        numeric_cols = df.select_dtypes(include=['number']).columns
        if numeric_cols.empty:
            fig.add_annotation(text="No numerical columns to plot against time", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

        numeric_col = numeric_cols[0]
        for col in columns:
            try:
                ts = pd.to_datetime(df[col], errors='coerce') if not pd.api.types.is_datetime64_any_dtype(df[col]) else df[col]
                tmp = pd.DataFrame({'time': ts, 'value': df[numeric_col]}).dropna().sort_values('time')
                if not tmp.empty:
                    fig.add_trace(go.Scatter(x=tmp['time'], y=tmp['value'], mode='lines+markers', name=f"{numeric_col} over {col}"))
            except Exception:
                continue

        if fig.data:
            fig.update_layout(title="Time Series Analysis", xaxis_title="Time", yaxis_title="Value", height=450)
        else:
            fig.add_annotation(text="Could not create time series plot with selected columns", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=450)
        return fig

    def _kmeans(self, data, k, max_iter=100, random_state=42):
        rng = np.random.default_rng(random_state)
        centroids = data.iloc[rng.choice(len(data), k, replace=False)].values
        for _ in range(max_iter):
            distances = np.linalg.norm(data.values[:, None] - centroids[None, :], axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([data.values[labels == i].mean(axis=0) for i in range(k)])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        return labels

    def _create_clustering_plot(self, df, columns, n_clusters=3, **kwargs):
        fig = go.Figure()
        num_df = df[columns].select_dtypes(include=['number']).dropna()
        if num_df.shape[1] < 2 or len(num_df) < n_clusters:
            fig.add_annotation(text="Need ≥2 numerical columns with enough rows for clustering", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=450)
            return fig
        labels = self._kmeans(num_df, k=min(n_clusters, len(num_df)))
        dims = num_df.shape[1]
        if dims >= 3:
            centered = num_df.values - num_df.values.mean(axis=0)
            u, s, vt = np.linalg.svd(centered, full_matrices=False)
            coords = u[:, :2] * s[:2]
            x, y = coords[:, 0], coords[:, 1]
            xlabel, ylabel = "PC1", "PC2"
        else:
            x, y = num_df.iloc[:, 0], num_df.iloc[:, 1]
            xlabel, ylabel = num_df.columns[0], num_df.columns[1]
        for c in range(min(n_clusters, len(num_df))):
            mask = labels == c
            fig.add_trace(go.Scatter(
                x=x[mask], y=y[mask], mode='markers',
                name=f"Cluster {c+1}", marker=dict(size=6)
            ))
        fig.update_layout(title="K-Means Clustering", xaxis_title=xlabel, yaxis_title=ylabel, height=450)
        return fig

    def _mutual_info(self, x, y, bins=10):
        xy = np.histogram2d(x, y, bins=bins)[0]
        xy = xy / xy.sum()
        xm = xy.sum(axis=1)
        ym = xy.sum(axis=0)
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if xy[i, j] > 0:
                    mi += xy[i, j] * np.log2(xy[i, j] / (xm[i] * ym[j] + 1e-10))
        return mi

    def _create_feature_importance_plot(self, df, columns, **kwargs):
        fig = go.Figure()
        num_df = df[columns].select_dtypes(include=['number']).dropna()
        if num_df.shape[1] < 2:
            fig.add_annotation(text="Need ≥2 numerical columns", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=450)
            return fig
        pairs = []
        for i in range(len(num_df.columns)):
            for j in range(i + 1, len(num_df.columns)):
                mi = self._mutual_info(num_df.iloc[:, i].values, num_df.iloc[:, j].values)
                pairs.append((f"{num_df.columns[i]} ↔ {num_df.columns[j]}", mi))
        pairs.sort(key=lambda x: -x[1])
        top = pairs[:10]
        names, scores = zip(*top) if top else ([], [])
        fig = go.Figure(go.Bar(x=scores, y=names, orientation='h', marker_color='#1f77b4'))
        fig.update_layout(title="Top Feature Pairs by Mutual Information", xaxis_title="Mutual Information (bits)", yaxis_title=None, height=400)
        return fig
