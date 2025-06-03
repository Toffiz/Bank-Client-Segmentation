# Customer Segmentation Project

This project implements advanced clustering techniques to segment bank customers based on their transaction behavior. The pipeline includes data preprocessing, feature engineering, and multiple clustering approaches (K-Means, UMAP+HDBSCAN, and Deep Embedded Clustering).


## Data Description

The dataset contains customer transaction records with the following attributes:
- Transaction timestamps
- Amounts (in KZT)
- MCC categories
- Transaction types
- POS entry modes
- Wallet types

Original dataset statistics:
- Total rows: [shown in notebook]
- Columns: 25+ features after processing

## Methodology

### 1. Feature Engineering

Key steps:
- Winsorization of transaction amounts (99th percentile)
- 365-day activity window
- RFM (Recency, Frequency, Monetary) features
- Channel preferences (transaction type distribution)
- Payment method preferences
- Spending category distributions
- 90-day ARPU trends

### 2. Clustering Approaches

#### A. K-Means
- Grid search over k=3-15 clusters
- StandardScaler normalization
- Evaluated with silhouette and Davies-Bouldin scores

#### B. UMAP + HDBSCAN
- Two-stage optimization:
  1. Fixed cluster range (10-25)
  2. Natural cluster discovery
- UMAP parameters tuned:
  - n_neighbors: 15-75
  - min_dist: 0-0.5
  - n_components: 10-20
- HDBSCAN parameters tuned:
  - min_cluster_size: 1-3% of data
  - min_samples: 5-10

#### C. Deep Embedded Clustering (DEC)
- Autoencoder architecture:
  - Encoder: 512-256-latent (20-100)
  - Decoder: symmetric
- Pretraining (20 epochs) + clustering (40 epochs)
- KL divergence + reconstruction loss
- Parameters tuned:
  - Latent dimensions: 20, 50, 100
  - Beta (loss weight): 0.05, 0.1, 0.2

## Results

Best performing models:
1. [Best K-Means] k=X | silhouette=0.XXX | db=0.XXX
2. [Best UMAP+HDBSCAN] clusters=X | sil=0.XXX | db=0.XXX
3. [Best DEC] latent=X | β=0.X | sil=0.XXX | db=0.XXX

Visualizations available for:
- 2D UMAP projections
- Cluster distributions
- Feature importance by cluster

## Requirements

Python 3.8+ with the following packages:
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
umap-learn>=0.5.0
hdbscan>=0.8.0
torch>=1.10.0
pyarrow>=6.0.0
matplotlib>=3.5.0
seaborn>=0.11.0


## Usage

1. Place `DECENTRATHON_3.0.parquet` in the project directory
2. Run the notebook sequentially:
   - Data loading and preprocessing
   - Feature engineering
   - Clustering experiments
   - Results analysis

## License

This project is licensed under the MIT License.

