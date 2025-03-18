

class TextFormatting():
    '''
        Text formatting mixin
    '''

    def autospace(self, elements_list, space, delimiter=' ') -> str:
        output = ''
        for i, element in enumerate(elements_list):
            if i == len(elements_list):
                output += element
            else:
                output += element + (space-len(element))*delimiter
        return output

    def print_groups(self, space=24, max_length=15):
        self.log('')
        # Get the number of groups (rows) and the maximum length of rows
        data = self.data
        num_groups = len(data)
        group_longest = max(len(row) for row in data)

        # Print the header
        header = [f'Group {i+1}' for i in range(num_groups)]
        line = [''*7]
        self.log(self.autospace(header, space))
        self.log(self.autospace(line, space))

        # Print each column with a placeholder if longer than max_length
        for i in range(group_longest):
            row_values = []
            all_values_empty = True
            for row in data:
                if len(row) > max_length:
                    if i < max_length:
                        row_values.append(str(row[i]))
                        all_values_empty = False
                    elif i == max_length:
                        row_values.append(f'[{len(row) - max_length} more]')
                        all_values_empty = False
                    else:
                        continue
                else:
                    if i < len(row):
                        row_values.append(str(row[i]))
                        all_values_empty = False
                    else:
                        row_values.append('')
            if all_values_empty:
                break
            self.log(self.autospace(row_values, space))

    def print_results(self):
        self.log('\n\nResults: \n')
        for i in self.results:
            shift = 27 - len(i)
            if i == 'Warnings':
                self.log(i, ':', ' ' * shift, len(self.results[i]))
            elif i == 'Posthoc_Tests_Name':
                self.log(i, ':', ' ' * shift,
                         self.results[i]) if self.results[i] != '' else 'N/A'
            elif i == 'Posthoc_Matrix':
                self.log(i, ':', ' ' * shift, '{0}x{0} matrix'.format(
                    len(self.results[i])) if self.results[i] else 'N/A')
            elif (i == 'Samples'
                  or i == 'Posthoc_Matrix_bool'
                  or i == 'Posthoc_Matrix_printed'
                  or i == 'Posthoc_Matrix_stars'
                  ):
                pass
            else:
                self.log(i, ':', ' ' * shift, self.results[i])

    def make_p_value_printed(self, p) -> str:
        if p is not None:
            if p > 0.99:
                return 'p>0.99'
            elif p >= 0.01:
                return f'p={p:.2g}'
            elif p >= 0.001:
                return f'p={p:.2g}'
            elif p >= 0.0001:
                return f'p={p:.1g}'
            elif p < 0.0001:
                return 'p<0.0001'
            else:
                return 'N/A'
        return 'N/A'

    def make_stars(self, p) -> int:
        if p is not None:
            if p < 0.0001:
                return 4
            if p < 0.001:
                return 3
            elif p < 0.01:
                return 2
            elif p < 0.05:
                return 1
            else:
                return 0
        return 0

    def make_stars_printed(self, n) -> str:
        return '*' * n if n else 'ns'
