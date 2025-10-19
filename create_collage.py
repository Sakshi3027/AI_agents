import matplotlib.pyplot as plt
from matplotlib.image import imread

# Read images
img1 = imread('outputs/distributions.png')
img2 = imread('outputs/correlation_heatmap.png')
img3 = imread('outputs/categorical_distributions.png')

# Create collage
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

axes[0].imshow(img1)
axes[0].axis('off')
axes[0].set_title('Distribution Analysis', fontsize=16, fontweight='bold')

axes[1].imshow(img2)
axes[1].axis('off')
axes[1].set_title('Correlation Heatmap', fontsize=16, fontweight='bold')

axes[2].imshow(img3)
axes[2].axis('off')
axes[2].set_title('Categorical Analysis', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('screenshots/analysis_collage.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Collage created: screenshots/analysis_collage.png")
