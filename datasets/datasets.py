import json
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from huggingface_hub import snapshot_download
import pathlib as pth
from pyarrow import parquet as pq
from pyarrow import Table as pqTable
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Callable

class DataConfig:
    name = 'wikitext_data'
    hf_repo_id = 'Salesforce/wikitext'
    hf_dataset_name = 'all'
    hf_pretrained_model = 'openai-community/gpt2'
    split: str | list[str] = 'train'

    root = pth.Path.home()
    data_storage_location = name +'/data'
    semi_data_storage_location = name +'/semi_data'
    token_storage_location = name + '/tokens'


class BaseDownloader:

    def download(self, *args, **kwargs):
        """
        Download a file from the internet/ huggingface repository.
        :param:
        :return:
        """
        raise NotImplementedError

    def run_data_to_semi_data(self, *args, **kwargs):
        """
        Load a config and set params.
        :return:
        """
        raise NotImplementedError


class ConfigManager:

    __allowed__ = [
        'dataset_repo_download',
        'dataset_folder_checkpoint',
        'dataset_folder_split_checkpoint',
        'dataset_semi_split_folder_checkpoint',
        'dataset_semi_split_folder_split_checkpoint',
        'dataset_token_folder_checkpoint',
        'dataset_token_folder_split_checkpoint',
        'dataset_token_last_iteration_checkpoint'
    ]

    def __init__(self, file_name: str):
        self._file_name = file_name

        if os.path.exists(file_name):
            with open(self._file_name, 'r') as f:
                self.config = json.load(f)
                for key, value in self.config.items():
                    setattr(self, key, value)

    def set_value(self, key: str, value: str | bool):
        setattr(self, key, value)
        # collection config
        with open(self._file_name, 'w') as f:
            json.dump(self.config, f)








class HFDataDownloader(BaseDownloader):
    """
    This class only follows parquet based files from HF basically bellow structure working only for text-data.
    - hf-repo
        - dataset_1
            - train
            - test
            - validation
            - etc.
        - dataset_2
            - train
            - test
            - validation
            - etc.

    """
    _adapter_struct = """
        def adapter(table: pqTable, *args, **kwargs):
            \"\"\"Thia function transform each row and yield it\"\"\"
    
    """
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._class = {}
        self._config_manager = ConfigManager('dataset_config.json')

    def config_manager(self, file_name, key, value):
        if os.path.isfile(file_name):
           with open(file_name, 'r') as f:
               config  = json.load(f)
        else:
            config = {}

        config[key] = value

        with open(file_name, 'w') as f:
            json.dump(config, f)



    def add_adapter_cls(self, func: Callable, name = 'default'):
        self._class[name] = func

    def process_printer(self, msg, logger_type='INFO'):
        if logger_type == 'INFO':
            self.logger.info(msg)
            print(msg)

    def split_table_to_parquet(self, table: pqTable, output_prefix, folder_path, chunk_size=100):
        n_rows = table.num_rows
        n_chunks = math.ceil(n_rows / chunk_size)

        for i in tqdm(range(n_chunks),desc=f'[INFO] {output_prefix}'):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_rows)
            chunk = table.slice(start, end - start)
            pq.write_table(chunk, f"{folder_path.strip()}/{output_prefix}-{i:05d}.parquet")

    def _tokenize_text(self, text: str, model: str):
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokens = tokenizer.encode(text)
        return tokens


    def file_tokenizer(self, files: list, source_dir: str, destination_dir: str, adapter_name: str, folder:str):
        os.makedirs(destination_dir, exist_ok=True)
        files.sort()
        for file in tqdm(files, desc=f'[INFO] {folder}'):
            file_name = file
            file = os.path.join(source_dir, file)
            table = pq.read_table(file)
            generator = self._class[adapter_name](table)
            tokens = []

            # Using ProcessPoolExecutor for parallel tokenization
            with ProcessPoolExecutor(max_workers=30) as executor:
                futures = {executor.submit(self._tokenize_text, row, self.config.hf_pretrained_model): row for row in
                           generator}
                for future in tqdm(as_completed(futures), total=len(futures), desc=f'[INFO] {file_name}'):
                    tokens.extend(future.result())

            file = os.path.basename(file)
            with open(os.path.join(destination_dir, f'{".".join(file.split('.')[:-1])}.json'), 'w') as f:
                f.write(json.dumps(tokens))

    def run_semi_data_to_tokens(self, semi_data_storage_location, token_storage_location, adapter_name):
        try:
            # Dataset folders
            _, folders, files = list(os.walk(semi_data_storage_location))[0]

            for folder in folders:
                self.process_printer(f'{folder}')
                dataset_folder = str(os.path.join(semi_data_storage_location, folder))
                token_dataset_folder = str(os.path.join(token_storage_location, folder))

                # Splits in dataset_folder
                _, split_folders, _ = list(os.walk(str(dataset_folder)))[0]
                for split_folder in split_folders:
                    destination_folder = os.path.join(token_dataset_folder, split_folder)
                    split_folder = os.path.join(dataset_folder, split_folder)

                    _, _, files = list(os.walk(str(split_folder)))[0]

                    # Handling each file from disk and convert into tokens

                    self.file_tokenizer(files, split_folder, destination_folder, adapter_name, folder)

        except Exception as e:
            print(e)

    def run_data_to_semi_data(self, data_location: pth.Path , semi_data_location: pth.Path):
        # reading datasets
        try:
            _, folders, files = list(os.walk(data_location))[0]

            for folder in folders:
                if folder.startswith('.'): continue

                if self.config.hf_dataset_name == 'all' or self.config.hf_dataset_name == folder:
                    source_folder = os.path.join(data_location, folder)
                    destination_folder = os.path.join(semi_data_location, folder)
                    os.makedirs(destination_folder, exist_ok=True)

                    # TODO: Add Logger Here
                    # We need to identify train, test, validation and other object from folder.
                    _, _, files = list(os.walk(source_folder))[0]
                    files.sort()
                    groups = {}
                    for file in files:
                        file_name, *_ = file.split('-')
                        file_name = file_name.strip()
                        groups[file_name] = {
                            'path': source_folder,
                            'files': groups.get(file_name, {}).get('files', []) + [file]
                        }

                    for split_type, file_info in groups.items():
                        split_files = [ os.path.join(source_folder, _) for _ in file_info['files'] ]
                        table = pq.read_table(split_files)

                        # Create Sub-folder for split type
                        split_path = os.path.join(destination_folder, split_type)
                        os.makedirs(split_path, exist_ok=True)
                        self.split_table_to_parquet(table, split_type, split_path, chunk_size=1000)

        except Exception as e:
            print(e)

    def download(self, adapter_name = 'default'):
        if not self._class.get(adapter_name):
            raise Exception(f'{adapter_name} not available. We need function that '
                            f'transform parquet file structure into formal text.')

        if not self._config_manager.dataset_repo_download:
            snapshot_download(self.config.hf_repo_id, repo_type='dataset',
                              local_dir=self.config.root.joinpath(self.config.data_storage_location))

        self._config_manager.set_value('dataset_repo_download', True)

        data_storage_location = self.config.root.joinpath(self.config.data_storage_location)
        token_storage_location = self.config.root.joinpath(self.config.token_storage_location)
        semi_data_storage_location = self.config.root.joinpath(self.config.semi_data_storage_location)

        # First we will convert data into semi_data each contain 100 rows max only
        self.run_data_to_semi_data(data_storage_location, semi_data_storage_location)
        self.run_semi_data_to_tokens(semi_data_storage_location, token_storage_location, adapter_name)



def pq_adapter(table: pqTable):
    table = table.to_pydict()
    table = table['text']
    for row in table:
        yield row

if __name__ == '__main__':
    hf_data_downloader = HFDataDownloader(DataConfig())
    hf_data_downloader.add_adapter_cls(pq_adapter)
    hf_data_downloader.download()