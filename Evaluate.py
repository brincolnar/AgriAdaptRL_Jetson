import os
import re
import pandas as pd
from Segment import Segment

class Evaluate(Segment):
    def numeric_sort(self, file_name):
        """Extract numeric value from file names for sorting."""
        numbers = re.findall(r'\d+', file_name)
        return int(numbers[0]) if numbers else 0

    def process_images(self):
        feature_results = []
        sorted_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')], key=self.numeric_sort)
        for file_name in sorted_files:
            results, probs, gt, filename = self.infer(file_name)
            print(f"Filename: {filename}, IoU: {results['test/iou/weeds']}")
            feature_results.append({
                'Filename': filename,
                'IoU': results['test/iou/weeds']
            })
        return feature_results

if __name__ == "__main__":
    networks = {
        '0%': 'unet_512_pruned_00_iterative_1.pt',
        '25%': 'unet_512_pruned_025_iterative_1.pt',
        '50%': 'unet_512_pruned_05_iterative_1.pt',
        '75%': 'unet_512_pruned_075_iterative_1.pt',
    }

    network_dir = 'alinas-models'

    config = {
        "image_dir": "./data/ordered_train_test/all/images/",
        "label_dir": "./data/ordered_train_test/all/labels/",
        "log_dir": f"./results/{network_dir}/"
    }

    performance_data = {}

    for prune_label, network_filename in networks.items():
        print(f"Evaluating network: {prune_label}")
        model_path = f'./garage/{network_filename}'
        evaluate_test = Evaluate(model_path=model_path, image_dir=config['image_dir'], label_dir=config['label_dir'], device='cuda')
        results = evaluate_test.process_images()

        for result in results:
            filename = result['Filename']
            iou = result['IoU']
            if filename not in performance_data:
                performance_data[filename] = {}
            performance_data[filename][prune_label] = iou

    df = pd.DataFrame.from_dict(performance_data, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Filename'}, inplace=True)

    output_csv_path = os.path.join(config['log_dir'], 'performance_results.csv')
    os.makedirs(config['log_dir'], exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print(f"Performance results saved to {output_csv_path}")
