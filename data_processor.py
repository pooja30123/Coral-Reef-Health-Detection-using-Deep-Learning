import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class CoralDataProcessor:
    def __init__(self, csv_path, images_dir, patch_size=64):
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.patch_size = patch_size
        
        # Define comprehensive coral health mapping
        self.health_mapping = {
            # Healthy coral species
            'acropora': 'healthy',
            'porites': 'healthy', 
            'pocillopora': 'healthy',
            'montipora': 'healthy',
            'pavona': 'healthy',
            'millepora': 'healthy',
            'astreopora': 'healthy',
            'leptastrea': 'healthy',
            'favites': 'healthy',
            'goniastrea': 'healthy',
            'echinopora': 'healthy',
            'platygyra': 'healthy',
            'favia': 'healthy',
            'fungia': 'healthy',
            'montastraea': 'healthy',
            'stylophora': 'healthy',
            'gardineroseris': 'healthy',
            'acanthastrea': 'healthy',
            'herpolitha': 'healthy',
            'leptoseris': 'healthy',
            'cyphastrea': 'healthy',
            'sandolitha': 'healthy',
            'lobophyllia': 'healthy',
            'psammocora': 'healthy',
            'soft_coral': 'healthy',
            
            # Dead/damaged coral
            'dead_coral': 'dead',
            'broken_coral_rubble': 'dead',
            
            # Bleached/stressed (algae indicates stress/bleaching)
            'algae': 'bleached',
            'green_fleshy_algae': 'bleached',
            'macroalgae': 'bleached',
            'turf': 'bleached',
            
            # Non-coral substrates (exclude from health analysis)
            'sand': 'substrate',
            'rock': 'substrate',
            'hard_substrate': 'substrate',
            'crustose_coralline_algae': 'substrate',
            'soft': 'substrate',
            
            # Invalid/unclear (exclude)
            'bad': 'invalid',
            'dark': 'invalid',
            'off': 'invalid',
            'tuba': 'invalid'
        }
        
        self.health_classes = ['healthy', 'bleached', 'dead']
        
    def load_annotations(self):
        """Load and process the CSV annotations"""
        print("Loading annotations...")
        df = pd.read_csv(self.csv_path)
        
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Clean data
        df = df.dropna(subset=['Name', 'Row', 'Column', 'Label'])
        
        # Map detailed labels to health categories
        df['health_label'] = df['Label'].map(self.health_mapping)
        
        # Filter out invalid/substrate labels for health classification
        health_df = df[df['health_label'].isin(self.health_classes)].copy()
        
        print(f"Total annotations: {len(df):,}")
        print(f"Health-relevant annotations: {len(health_df):,}")
        print(f"Unique images: {df['Name'].nunique():,}")
        
        # Show detailed label distribution
        print("\nOriginal label distribution (top 15):")
        label_counts = df['Label'].value_counts().head(15)
        for label, count in label_counts.items():
            print(f"  {label}: {count:,}")
        
        print("\nHealth label distribution:")
        health_counts = health_df['health_label'].value_counts()
        for label, count in health_counts.items():
            print(f"  {label}: {count:,}")
        
        return df, health_df
    
    def extract_image_patches(self, df, save_dir='data/patches', balance_classes=True, max_per_class=5000):
        """Extract image patches around annotation points"""
        os.makedirs(save_dir, exist_ok=True)
        
        patches = []
        labels = []
        image_names = []
        coordinates = []
        
        # Balance classes if requested
        if balance_classes:
            min_count = min(df['health_label'].value_counts())
            max_samples = min(max_per_class, min_count)
            
            print(f"Balancing classes with {max_samples:,} samples each...")
            balanced_df = pd.DataFrame()
            for health_class in self.health_classes:
                class_df = df[df['health_label'] == health_class]
                if len(class_df) > max_samples:
                    class_df = class_df.sample(n=max_samples, random_state=42)
                balanced_df = pd.concat([balanced_df, class_df])
            df = balanced_df.reset_index(drop=True)
        
        print(f"Extracting patches for {len(df):,} annotations...")
        
        half_patch = self.patch_size // 2
        successful_extractions = 0
        failed_extractions = 0
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processed {idx:,}/{len(df):,} annotations...")
                
            image_path = os.path.join(self.images_dir, row['Name'])
            
            if not os.path.exists(image_path):
                failed_extractions += 1
                continue
                
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    failed_extractions += 1
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w = image.shape[:2]
                
                # Get patch coordinates
                row_coord = int(row['Row'])
                col_coord = int(row['Column'])
                
                # Check bounds
                if (row_coord - half_patch < 0 or row_coord + half_patch >= h or
                    col_coord - half_patch < 0 or col_coord + half_patch >= w):
                    failed_extractions += 1
                    continue
                
                # Extract patch
                patch = image[row_coord-half_patch:row_coord+half_patch,
                             col_coord-half_patch:col_coord+half_patch]
                
                if patch.shape[:2] != (self.patch_size, self.patch_size):
                    failed_extractions += 1
                    continue
                
                patches.append(patch)
                labels.append(row['health_label'])
                image_names.append(row['Name'])
                coordinates.append((row_coord, col_coord))
                successful_extractions += 1
                
            except Exception as e:
                print(f"Error processing {row['Name']}: {e}")
                failed_extractions += 1
                continue
        
        print(f"Successfully extracted {successful_extractions:,} patches")
        print(f"Failed extractions: {failed_extractions:,}")
        
        # Convert to numpy arrays
        patches = np.array(patches)
        labels = np.array(labels)
        
        # Show final class distribution
        print("\nFinal patch distribution:")
        label_counts = Counter(labels)
        for label, count in label_counts.items():
            print(f"  {label}: {count:,}")
        
        # Save processed data
        data_dict = {
            'patches': patches,
            'labels': labels,
            'image_names': image_names,
            'coordinates': coordinates,
            'health_classes': self.health_classes,
            'patch_size': self.patch_size
        }
        
        with open(os.path.join(save_dir, 'processed_data.pkl'), 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"Saved processed data to {save_dir}/processed_data.pkl")
        return patches, labels, image_names, coordinates
    
    def create_train_test_split(self, patches, labels, test_size=0.2, val_size=0.2):
        """Create train/validation/test splits"""
        print("Creating train/validation/test splits...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            patches, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"Train set: {len(X_train):,} samples")
        print(f"Validation set: {len(X_val):,} samples") 
        print(f"Test set: {len(X_test):,} samples")
        
        # Encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(self.health_classes)
        
        y_train_encoded = label_encoder.transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Show class distribution in each set
        print("\nClass distribution in splits:")
        for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            counts = Counter(y_split)
            print(f"{split_name}: {dict(counts)}")
        
        return (X_train, X_val, X_test, 
                y_train_encoded, y_val_encoded, y_test_encoded,
                label_encoder)
    
    def visualize_dataset(self, df):
        """Create comprehensive visualizations of the dataset"""
        print("Creating dataset visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Coral Reef Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Health label distribution
        health_counts = df['health_label'].value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # green, orange, red
        axes[0, 0].pie(health_counts.values, labels=health_counts.index, autopct='%1.1f%%', 
                      colors=colors, startangle=90)
        axes[0, 0].set_title('Health Label Distribution', fontweight='bold')
        
        # 2. Original label distribution (top 15)
        top_labels = df['Label'].value_counts().head(15)
        axes[0, 1].barh(range(len(top_labels)), top_labels.values, color='skyblue')
        axes[0, 1].set_yticks(range(len(top_labels)))
        axes[0, 1].set_yticklabels(top_labels.index, fontsize=9)
        axes[0, 1].set_title('Top 15 Original Labels', fontweight='bold')
        axes[0, 1].set_xlabel('Count')
        
        # 3. Annotations per image distribution
        img_counts = df['Name'].value_counts()
        axes[0, 2].hist(img_counts, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 2].set_xlabel('Annotations per Image')
        axes[0, 2].set_ylabel('Number of Images')
        axes[0, 2].set_title('Annotations per Image Distribution', fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Health labels by image (stacked bar)
        health_by_image = df.groupby(['Name', 'health_label']).size().unstack(fill_value=0)
        health_ratios = health_by_image.div(health_by_image.sum(axis=1), axis=0)
        
        sample_images = health_ratios.head(20)  # Show first 20 images
        sample_images.plot(kind='bar', stacked=True, ax=axes[1, 0], 
                          color=['#2ecc71', '#f39c12', '#e74c3c'])
        axes[1, 0].set_title('Health Composition (First 20 Images)', fontweight='bold')
        axes[1, 0].set_xlabel('Image Name')
        axes[1, 0].set_ylabel('Proportion')
        axes[1, 0].legend(title='Health Status')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Correlation between image size and annotation count
        image_stats = []
        for image_name in df['Name'].unique()[:100]:  # Sample first 100 images
            image_path = os.path.join(self.images_dir, image_name)
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    width, height = img.size
                    area = width * height
                    ann_count = len(df[df['Name'] == image_name])
                    image_stats.append({'area': area, 'annotations': ann_count})
                except:
                    continue
        
        if image_stats:
            stats_df = pd.DataFrame(image_stats)
            axes[1, 1].scatter(stats_df['area'], stats_df['annotations'], alpha=0.6, color='purple')
            axes[1, 1].set_xlabel('Image Area (pixels)')
            axes[1, 1].set_ylabel('Number of Annotations')
            axes[1, 1].set_title('Image Size vs Annotation Count', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Spatial distribution of annotations (sample image)
        sample_image = df['Name'].iloc[0]
        sample_data = df[df['Name'] == sample_image]
        
        if len(sample_data) > 10:  # Only if enough points
            axes[1, 2].scatter(sample_data['Column'], sample_data['Row'], 
                             c=[colors[self.health_classes.index(label)] for label in sample_data['health_label']], 
                             alpha=0.7, s=20)
            axes[1, 2].set_xlabel('Column (X)')
            axes[1, 2].set_ylabel('Row (Y)')
            axes[1, 2].set_title(f'Spatial Distribution\n({sample_image})', fontweight='bold')
            axes[1, 2].invert_yaxis()  # Invert Y to match image coordinates
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def sample_patches_visualization(self, patches, labels, n_samples=15):
        """Visualize sample patches for each class"""
        print("Creating sample patches visualization...")
        
        fig, axes = plt.subplots(len(self.health_classes), n_samples//len(self.health_classes), 
                                figsize=(18, 8))
        
        colors = {'healthy': '#2ecc71', 'bleached': '#f39c12', 'dead': '#e74c3c'}
        
        for i, health_class in enumerate(self.health_classes):
            class_indices = np.where(labels == health_class)[0]
            if len(class_indices) == 0:
                continue
                
            n_class_samples = min(n_samples//len(self.health_classes), len(class_indices))
            sample_indices = np.random.choice(class_indices, size=n_class_samples, replace=False)
            
            for j, idx in enumerate(sample_indices):
                if len(self.health_classes) == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j] if axes.ndim > 1 else axes[i]
                    
                ax.imshow(patches[idx])
                ax.set_title(f'{health_class.title()}', 
                           color=colors[health_class], fontweight='bold')
                ax.axis('off')
                
                # Add a colored border
                for spine in ax.spines.values():
                    spine.set_edgecolor(colors[health_class])
                    spine.set_linewidth(3)
        
        plt.suptitle('Sample Patches by Health Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('sample_patches.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to process the coral reef dataset"""
    print("="*60)
    print("CORAL REEF HEALTH DETECTION - DATA PROCESSING")
    print("="*60)
    
    # Initialize processor
    processor = CoralDataProcessor(
        csv_path='data/raw/combined_annotations_remapped.csv',
        images_dir=r'C:\Users\Susanta Baidya\Desktop\coral-reef-detection\data\raw\images\images',
        patch_size=64
    )
    
    # Create directories
    os.makedirs('data/patches', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Load and process annotations
    df_all, df_health = processor.load_annotations()
    
    # Visualize dataset
    processor.visualize_dataset(df_health)
    
    # Extract patches
    patches, labels, image_names, coordinates = processor.extract_image_patches(
        df_health, 
        save_dir='data/patches',
        balance_classes=True, 
        max_per_class=3000
    )
    
    if len(patches) == 0:
        print("ERROR: No patches were extracted. Please check your image paths and CSV file.")
        return
    
    # Visualize sample patches
    processor.sample_patches_visualization(patches, labels)
    
    # Create train/test splits
    splits = processor.create_train_test_split(patches, labels)
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = splits
    
    print("\n" + "="*60)
    print("DATA PROCESSING COMPLETE!")
    print("="*60)
    print(f"✓ Total patches extracted: {len(patches):,}")
    print(f"✓ Classes: {list(label_encoder.classes_)}")
    print(f"✓ Train set: {len(X_train):,} samples")
    print(f"✓ Validation set: {len(X_val):,} samples")
    print(f"✓ Test set: {len(X_test):,} samples")
    print("✓ Data saved to: data/patches/processed_data.pkl")
    print("\nNext step: Run model training with: python model/train_model.py")
    
    return processor, splits

if __name__ == "__main__":
    main()