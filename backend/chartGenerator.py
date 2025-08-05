import base64
import io
import logging
import os
import uuid

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from backend import config

logger = logging.getLogger(__name__)


class ChartGenerator:
    """
    Generates and saves charts from a given DataFrame based on a user query.
    """

    def __init__(self, chart_dir: str = config.CHART_DIR):
        """
        Initializes the ChartGenerator.

        Args:
            chart_dir (str): The directory where generated charts will be saved.
        """
        self.chart_dir = chart_dir
        os.makedirs(self.chart_dir, exist_ok=True)

    def generate_chart(self, df: pd.DataFrame, query: str) -> str:
        """
        Analyzes a query to determine the chart type and generates it.

        Args:
            df (pd.DataFrame): The DataFrame to generate the chart from.
            query (str): The user's query describing the desired chart.

        Returns:
            The file path of the generated chart image.
        """
        logger.info(f"Generating chart for query: '{query}'")
        try:
            # Set the style for better-looking charts
            sns.set_style("whitegrid")
            
            # First, check for categorical data in the query
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            query_words = query.lower().split()
            
            # Check if any categorical column is mentioned in the query
            for col in categorical_cols:
                if col.lower() in query_words:
                    logger.info(f"Detected categorical column: {col}")
                    return self._create_categorical_plot(df, col)
            
            # If no categorical data is found, proceed with regular chart types
            if "bar" in query.lower():
                chart_path = self._create_bar_chart(df, query)
            elif "line" in query.lower():
                chart_path = self._create_line_chart(df, query)
            elif "scatter" in query.lower():
                chart_path = self._create_scatter_plot(df, query)
            else:
                # Try to intelligently determine the best chart type
                logger.info("Analyzing data to determine the best visualization")
                chart_path = self._create_smart_plot(df, query)

            logger.info(f"Successfully generated chart: {chart_path}")
            return chart_path
        except Exception as e:
            logger.error(f"Failed to generate chart for query '{query}': {e}")
            raise

    def _create_bar_chart(self, df: pd.DataFrame, query: str) -> str:
        """Creates a bar chart."""
        # Simplified: assumes the first two columns are suitable for a bar chart.
        # An advanced version would parse column names from the query.
        x_col, y_col = df.columns[0], df.columns[1]
        plt.figure(figsize=(10, 6))
        df.plot(kind="bar", x=x_col, y=y_col)
        plt.title(f"Bar Chart: {y_col} by {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.xticks(rotation=45)
        return self._save_chart()

    def _create_line_chart(self, df: pd.DataFrame, query: str) -> str:
        """Creates a line chart."""
        x_col, y_col = df.columns[0], df.columns[1]
        plt.figure(figsize=(10, 6))
        df.plot(kind="line", x=x_col, y=y_col)
        plt.title(f"Line Chart: {y_col} over {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        return self._save_chart()

    def _create_scatter_plot(self, df: pd.DataFrame, query: str) -> str:
        """Creates a scatter plot."""
        x_col, y_col = df.columns[0], df.columns[1]
        plt.figure(figsize=(10, 6))
        df.plot(kind="scatter", x=x_col, y=y_col)
        plt.title(f"Scatter Plot: {y_col} vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        return self._save_chart()

    def _create_categorical_plot(self, df: pd.DataFrame, category_col: str) -> str:
        """Creates a plot for categorical data using seaborn."""
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x=category_col)
        plt.title(f"Distribution of {category_col}")
        plt.xticks(rotation=45)
        return self._save_chart()

    def _create_smart_plot(self, df: pd.DataFrame, query: str) -> str:
        """Intelligently determines and creates the most appropriate plot type."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) >= 2:
            # If we have multiple numeric columns, create a correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
            plt.title("Correlation Heatmap")
        elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
            # If we have one numeric and one categorical, create a box plot
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df, x=categorical_cols[0], y=numeric_cols[0])
            plt.title(f"{numeric_cols[0]} by {categorical_cols[0]}")
            plt.xticks(rotation=45)
        else:
            # Fallback to generic plot
            return self._create_generic_plot(df)
        
        return self._save_chart()

    def _create_generic_plot(self, df: pd.DataFrame) -> str:
        """Creates a generic plot from the first numeric column."""
        numeric_cols = df.select_dtypes(include="number").columns
        if not numeric_cols.empty:
            col_to_plot = numeric_cols[0]
            plt.figure(figsize=(10, 6))
            df[col_to_plot].plot(kind="line")
            plt.title(f"Generic Plot of {col_to_plot}")
            plt.xlabel("Index")
            plt.ylabel(col_to_plot)
            return self._save_chart()
        else:
            raise ValueError("No numeric columns found for generic plot.")

    def _save_chart(self) -> str:
        """Saves the current matplotlib figure to a file."""
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(self.chart_dir, filename)
        
        plt.savefig(filepath, format="png", bbox_inches="tight")
        plt.close()
        
        # Return the web-accessible path
        return f"/charts/{filename}"
