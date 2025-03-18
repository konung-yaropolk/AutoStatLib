import numpy as np
import pandas as pd


class Helpers():

    def matrix_to_dataframe(self, matrix):
        data = []
        cols = []
        rows = []

        order_number = 1
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                data.append(value)
                cols.append(i)
                rows.append(j)
                order_number += 1

        df = pd.DataFrame(
            {'Row': rows, 'Col': cols, 'Value': data})
        return df

    def floatify_recursive(self, data):
        if isinstance(data, list):
            # Recursively process sublists and filter out None values
            processed_list = [self.floatify_recursive(item) for item in data]
            return [item for item in processed_list if item is not None]
        else:
            try:
                # Try to convert the item to float
                return np.float64(data)
            except (ValueError, TypeError):
                # If conversion fails, replace with None
                self.warning_flag_non_numeric_data = True
                return None

    def create_results_dict(self) -> dict:

        self.stars_int = self.make_stars(self.p_value.item())
        self.stars_str = self.make_stars_printed(self.stars_int)

        return {
            'p-value': self.make_p_value_printed(self.p_value.item()),
            'Significance(p<0.05)':  True if self.p_value.item() < 0.05 else False,
            'Stars_Printed': self.stars_str,
            'Test_Name': self.test_name,
            'Groups_Compared': self.n_groups,
            'Population_Mean': self.popmean if self.n_groups == 1 else 'N/A',
            'Data_Normaly_Distributed': self.parametric,
            'Parametric_Test_Applied': True if self.test_id in self.test_ids_parametric else False,
            'Paired_Test_Applied': self.paired,
            'Tails': self.tails,
            'p-value_exact': self.p_value.item(),
            'Stars':  self.stars_int,
            # 'Stat_Value': self.test_stat.item(),
            'Warnings': self.warnings,
            'Groups_N': [len(self.data[i]) for i in range(len(self.data))],
            'Groups_Median': [np.median(self.data[i]).item() for i in range(len(self.data))],
            'Groups_Mean': [np.mean(self.data[i]).item() for i in range(len(self.data))],
            'Groups_SD': [np.std(self.data[i]).item() for i in range(len(self.data))],
            'Groups_SE': [np.std(self.data[i]).item() / np.sqrt(len(self.data)).item() for i in range(len(self.data))],
            # actually returns list of lists of numpy dtypes of float64, next make it return regular floats:
            'Samples': self.data,
            'Posthoc_Tests_Name': self.posthoc_name if self.posthoc_name is not None else '',
            'Posthoc_Matrix': self.posthoc_matrix if self.posthoc_matrix else [],
            'Posthoc_Matrix_bool': [[bool(element) for element in row] for row in self.posthoc_matrix] if self.posthoc_matrix else [],
            'Posthoc_Matrix_printed': [[self.make_p_value_printed(element) for element in row] for row in self.posthoc_matrix] if self.posthoc_matrix else [],
            'Posthoc_Matrix_stars': [[self.make_stars_printed(self.make_stars(element)) for element in row] for row in self.posthoc_matrix] if self.posthoc_matrix else [],
        }

    def log(self, *args, **kwargs):
        message = ' '.join(map(str, args))
        # print(message, **kwargs)
        self.summary += '\n' + message

    def AddWarning(self, warning_id):
        message = self.warning_ids_all[warning_id]
        self.log(message)
        self.warnings.append(message)
