from pathlib import Path
from tqdm import tqdm
import shutil
from ruamel.yaml import YAML

def generate_configs(base_path: Path, positions: list, K_values: list, yaml: YAML =None):
    """
    Generates configurations for layer pruning by copying a base configuration and updating it with new values of K and position.
    
    Args:
        base_path (Path): The path to the base configuration directory.
        positions (list): List of positions ('top', 'bottom') to apply layer pruning.
        K_values (list): List of integers specifying the number of layers to drop.
    """
    index = 16  # Starting index for folder naming
    pbar = tqdm(K_values, desc="Generating configurations")
    for K in pbar:
        _pbar = tqdm(positions, leave=False)
        for position in _pbar:
            # Create new folder name
            new_folder_name = f"v{index}_layer_pruning_{K}_{position}"
            _pbar.set_description(f"Processing {new_folder_name}")
            new_folder_path = base_path.parent / new_folder_name
            
            # Copy base folder content to new folder
            shutil.copytree(base_path, new_folder_path, dirs_exist_ok=True)
            
            # Load and update config file
            config_file_path = new_folder_path / "config.yaml"
            config_content = config_file_path.read_text(encoding='utf-8')
            config = yaml.load(config_content)
            
            # Update the layer pruning settings
            config['layer_pruning']['enabled'] = True
            config['layer_pruning']['num_layers'] = K
            config['layer_pruning']['position'] = position
            
            # Save updated config back to file
            with open(config_file_path, 'w') as f:
                yaml.dump(config, f)

            trailing_blanks = ''.join(config_content.rsplit('\n', 1)[-1:])
            if trailing_blanks.isspace():  # If only whitespace exists after last \n
                with config_file_path.open('a', encoding='utf-8') as f:
                    f.write(trailing_blanks)  # Append trailing whitespace
            
            index += 1  # Increment index for next folder name

if __name__ == "__main__":
    # Configure YAML parser for max preservation
    yaml = YAML()
    yaml.preserve_quotes = True   # Keep original quoting style
    yaml.width = 4096             # Prevent line wrapping
    yaml.indent(mapping=2, sequence=4, offset=2)  # Match common indentation
    MIN = 0  
    MAX = 22
    STEP = 4  # Step size for K values
    base_path = Path("variants/v14_layer_pruning")
    positions = ['top', 'bottom']
    K_values = list(range(MIN, MAX, STEP))  # Generates values from 0 to 21 inclusive with step of 4
    generate_configs(base_path, positions, K_values[1:], yaml) # K_values[1:] otherwise 0 is included
