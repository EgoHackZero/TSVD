# TSVD - Car Accidents Analysis with PySpark

A comprehensive data mining and machine learning project analyzing UK car accident data using Apache Spark and PySpark. This project demonstrates end-to-end data science workflow including data integration, preprocessing, clustering, and classification modeling.

## Project Overview

This project performs in-depth analysis of car accident data to predict accident severity and identify patterns. It includes:

- **Data Integration**: Combining multiple datasets (Accidents, Casualties, Vehicles) with weather information
- **Data Preprocessing**: Feature engineering, transformation, and statistical analysis
- **Descriptive Modeling**: K-Means clustering for pattern discovery and anomaly detection
- **Predictive Modeling**: Multiple classification algorithms for accident severity prediction
- **Model Evaluation**: Comprehensive performance metrics and visualization

## Features

### 1. Data Integration (3 points)
- Integration of multiple CSV datasets (Accidents, Casualties, Vehicles)
- Stratified sampling while preserving target attribute distribution (10% sample)
- Train-test split (70/30 ratio)

### 2. Data Preprocessing (7 points)
- **Feature Transformation**:
  - Nominal to numerical encoding (StringIndexer)
  - Numerical to nominal binning (QuantileDiscretizer)
  - One-hot encoding for categorical features
- **Feature Selection**:
  - Information gain calculation using Chi-Square test
  - Top 10 most informative features identification
- **Statistical Analysis**:
  - Descriptive statistics for numerical attributes
  - Correlation matrix computation and visualization
  - Histogram generation for nominal attributes
- **Missing Value Handling**: Data cleaning and imputation

### 3. Descriptive Modeling (3 points)
- **K-Means Clustering**:
  - Optimal cluster selection (k=3-9) using Silhouette score
  - Best model: k=4 with score 0.651
- **Anomaly Detection**:
  - Distance-based outlier detection
  - 98th percentile threshold for anomaly identification

### 4. Classification Models (4 points)

Implementation of multiple classification algorithms:
- Decision Tree Classifier
- Naive Bayes (Multinomial)
- Random Forest Classifier (100 trees)
- Gradient-Boosted Trees (50 iterations)
- Linear SVM with One-vs-Rest strategy

### 5. Model Evaluation (3 points)

Comprehensive evaluation metrics:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix visualization

## Dataset

The project uses the UK Car Accidents dataset containing:
- **Accidents.csv**: Accident details (location, time, conditions)
- **Casualties.csv**: Casualty information (age, severity, type)
- **Vehicles.csv**: Vehicle details (type, maneuver, driver info)

Key features analyzed:
- Number of vehicles and casualties
- Speed limit
- Weather and light conditions
- Road surface conditions
- Junction details
- Driver age
- Urban/rural area classification

## Technical Stack

- **Apache Spark**: Distributed data processing
- **PySpark**: Python API for Spark
  - `pyspark.sql`: DataFrames and SQL operations
  - `pyspark.ml`: Machine learning pipeline
  - `pyspark.mllib`: Statistical functions
- **Python Libraries**:
  - `matplotlib`: Visualization
  - `seaborn`: Statistical data visualization
  - `pandas`: Data manipulation
  - `scikit-learn`: Additional metrics (MCC, confusion matrix)

## Installation

### Prerequisites

1. **Java 8 or 11** (required for Spark)
2. **Python 3.7+**
3. **Apache Spark 3.x**

### Setup

1. Install Spark:
```bash
# Download Spark from https://spark.apache.org/downloads.html
# Extract and set SPARK_HOME environment variable
```

2. Install Python dependencies:
```bash
pip install pyspark findspark matplotlib seaborn pandas scikit-learn
```

3. Configure Spark (optional - adjust memory settings):
```python
spark = SparkSession.builder \
    .appName("CarAccidents") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
```

## Usage

### Running the Analysis

1. **Prepare the dataset**:
   - Place CSV files in `../datalab/TSVD/dataset/CarAccidents/` directory:
     - `Accidents.csv`
     - `Casualties.csv`
     - `Vehicles.csv`

2. **Execute the notebook**:
   ```bash
   jupyter notebook main.ipynb
   ```

3. **Run cells sequentially** to perform:
   - Data loading and integration
   - Preprocessing and feature engineering
   - Statistical analysis and visualization
   - Model training and evaluation

### Key Functions

#### Data Integration
```python
read_csv_files(spark, file_paths, on="id", how="inner")
sample_by_percent(df, label, percent)
```

#### Preprocessing
```python
preprocess_data(df, cols)  # Returns processed dataframes
calculate_information_gain(df, cat_cols, target_col)
calculate_numerical_stats(df, numeric_cols)
create_histograms(df, nominal_cols)
```

#### Modeling
```python
fit_kmeans(k, evaluator, train, test)
fit_and_eval(estimator, name)  # Train and evaluate classifier
```

## Results

### Top Informative Features (by Information Gain)
1. Casualty Type (0.0187)
2. Vehicle Type (0.0089)
3. Vehicle Manoeuvre (0.0061)
4. Speed Limit (0.0055)
5. Local Authority District (0.0029)

### Clustering Performance
- Best K-Means model: k=4
- Silhouette Score: 0.651
- Anomaly threshold: 573.02 (98th percentile distance)

### Classification Performance
Models are evaluated on stratified test set with balanced classes:
- Accuracy, Precision, Recall, F1-score
- Matthews Correlation Coefficient (MCC)
- Confusion matrices for detailed error analysis

## Project Structure

```
TSVD/
├── main.ipynb              # Main Jupyter notebook with complete analysis
├── README.md               # Project documentation (this file)
├── LICENSE                 # MIT License
├── .gitignore             # Git ignore rules
└── dataset/               # Data directory (not included in repo)
    └── CarAccidents/
        ├── Accidents.csv
        ├── Casualties.csv
        └── Vehicles.csv
```

## Data Handling

### Undersampling Strategy
To handle class imbalance, the project implements undersampling:
```python
undersample_class(data)  # Balance classes to minority class size
```

### Feature Engineering Pipeline
1. Quantile discretization for numerical features
2. StringIndexer for categorical encoding
3. OneHotEncoder for nominal features
4. VectorAssembler for feature vectorization

## Performance Considerations

- **Partitioning**: Data is repartitioned to 20 partitions for optimal parallelism
- **Caching**: Intermediate results are cached to avoid recomputation
- **Memory**: Large datasets may require increasing JVM heap space

## Known Issues

- Random Forest may fail with `OutOfMemoryError` on large datasets
  - Solution: Reduce `numTrees` or increase Spark memory allocation
- Notebook includes Slovak language comments (data mining course requirement)

## Future Improvements

- [ ] Hyperparameter tuning using cross-validation
- [ ] Feature importance analysis
- [ ] Time-series analysis of accident trends
- [ ] Geographic visualization of accident hotspots
- [ ] Deep learning models (MLlib or TensorFlow integration)
- [ ] Real-time prediction API deployment

## Academic Context

This project was developed as part of a data mining course (TSVD - Techniky spracovania veľkých dát) demonstrating:
- Large-scale data processing with Spark
- End-to-end machine learning workflow
- Statistical analysis and visualization
- Model evaluation and comparison

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2024 EgoHackZero. All rights reserved.

## Author

**EgoHackZero**

## Acknowledgments

- UK Department for Transport for the Car Accidents dataset
- Apache Spark community for excellent documentation
- Course instructors for project guidance

## Contributing

This is an academic project. If you find issues or have suggestions, feel free to open an issue or submit a pull request.

## References

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [PySpark ML Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [UK Road Safety Data](https://data.gov.uk/dataset/road-accidents-safety-data)
