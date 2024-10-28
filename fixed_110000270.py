import numpy as np

def encode_hamming_7_4(x: np.ndarray):
    """
        Encodes a 4-bit input array using the Hamming (7,4) code.
        
        Parameters:
            x (np.ndarray): A 4-bit numpy array to be encoded. The array should have a size of 4.
        Returns:
            np.ndarray: A 7-bit numpy array representing the encoded input.
        Raises:
            ValueError: If the input array does not have a size of 4.
        Example:
            >>> encode_hamming_7_4(np.array([1, 0, 1, 1]))
            array([1, 0, 1, 1, 0, 1, 0])
    """
    
    
    # x should be a 4-bit array
    if x.size != 4:
        raise ValueError("Input should be a 4-bit array, but the size of x is {}".format(x.shape))
    
    # [TODO] Implement this function
    # Hint: What is the relationship between the special addition (e.g. 0+1=1, 1+1=0 defined in the problem)
    # and the modulo operation
    A = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1]
    ])
    
    
    return np.dot(A, x) % 2

# Test your implementation
x = np.array([1, 0, 1, 1])
encoded_x = encode_hamming_7_4(x)

print("Original data (x): ", x)
print("Hamming code of data (encoded_x): ", encoded_x)

# Some helper functions for creating bit array and adding single bit error

def convert_bit_array_to_btyes(bit_array: np.ndarray):
    """
        Convert a bit array to bytes.
        This function takes an array of bits (containing only 0s and 1s) and converts it into a byte array.
        If the length of the bit array is not a multiple of 8, it will be padded with zeros at the end to make it a multiple of 8.
        
        Parameters:
            bit_array (numpy.ndarray): A numpy array containing bits (0s and 1s).
        Returns:
            bytes: A byte array representation of the input bit array.
    """
    
    # bit array contains only 0 and 1
    # If the length of bit_array is not a multiple of 8, it will be padded with zeros at the end
    return np.packbits(bit_array).tobytes()

def create_single_bit_error():
    """
        Generates a single-bit error vector for a 7-bit codeword.
        This function creates a 7-element numpy array initialized to zeros.
        It then randomly selects one of the 7 positions and sets it to 1,
        simulating a single-bit error in a 7-bit codeword.
        
        Returns:
            numpy.ndarray: A 7-element array with a single bit set to 1 and the rest set to 0.
    """

    error = np.zeros(7)
    error[np.random.randint(7)] = 1
    return error

def add_single_bit_error(x: np.ndarray, error_prob: float=0.5):
    """
        Introduces a single-bit error to a 7-bit array with a given probability.
        
        Parameters:
            x (np.ndarray): A 7-bit numpy array to which the error will be added.
            error_prob (float): The probability of introducing a single-bit error. Default is 0.5.
        Returns:
            np.ndarray: The 7-bit array with a single-bit error introduced based on the given probability.
        Raises:
            ValueError: If the input array is not a 7-bit array.
    """
    
    # x should be a 4-bit array
    if x.size != 7:
        raise ValueError("Input should be a 7-bit array, but the size of x is {}".format(x.shape))
    
    if np.random.rand() < error_prob:
        return (x + create_single_bit_error()) % 2
    else:
        return x
    
def check_hamming_7_4(y: np.ndarray):
    """
        Checks the Hamming (7,4) code for a given 7-bit array.
        This function takes a 7-bit array as input and returns a 3-bit array that represents 
        the result of the Hamming (7,4) parity check matrix multiplied by the input array.
        The result can be used to detect and correct single-bit errors in the input array.
    
        Parameters:
            y (np.ndarray): A 7-bit numpy array representing the input codeword.
        Returns:
            np.ndarray: A 3-bit numpy array representing the result of the parity check.
        Raises:
            ValueError: If the input array is not 7 bits in size.
    """
    
    
    # x should be a 7-bit array
    if y.size != 7:
        raise ValueError("Input should be a 7-bit array, but the size of x is {}".format(x.shape))
    
    # [TODO] Implement this function by returning the 3-bit array Hy
    # Hint: What is the relationship between the special addition (e.g. 0+1=1, 1+1=0 defined in the problem)
    # and the modulo operation?
    
    H = np.array([
        [1, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 1]
    ])
    
    return np.dot(H, y) % 2

def fix_hamming_7_4(y: np.ndarray):
    """
        Corrects a 7-bit Hamming (7,4) encoded array and returns the corrected 4-bit data.
    
        Parameters:
            y (np.ndarray): A 7-bit numpy array representing the Hamming (7,4) encoded data.
        Returns:
            np.ndarray: A 4-bit numpy array representing the corrected data.
        Raises:
            ValueError: If the input array is not of size 7.
        Notes:
            This function assumes that the input array `y` is a Hamming (7,4) encoded array.
            It detects and corrects a single-bit error in the 7-bit array, if present, and
            returns the corrected 4-bit data.
    """
    
    # x should be a 7-bit array
    if y.size != 7:
        raise ValueError("Input should be a 7-bit array, but the size of x is {}".format(y.shape))
    
    # Make a copy of the input array to avoid modifying the original array
    y = y.copy()
    
    # Compute the 3-bit array Hy
    error_indicator = check_hamming_7_4(y).astype(int)
    
    # [TODO] Find out where is the error, leave it as -1 if there is no error
    error_sum = error_indicator[0] * 4 + error_indicator[1] * 2 + error_indicator[2] * 1
    error_index = error_sum - 1 if error_sum > 0 else -1
    
    # Correct it if there is an error
    if error_index != -1:
        # [TODO] Correct that bit
        y[error_index] = (y[error_index] + 1) % 2
    
    return y[:4]

# Test the functions
x = np.array([1, 0, 1, 1])
y = encode_hamming_7_4(x)
corrupt_y = add_single_bit_error(y)
corrected_x = fix_hamming_7_4(corrupt_y)

print("Original data (x): ", x)
print("Hamming code (y): ", y)
print("Corrupted data (corrupt_y): ", corrupt_y)
print("Corrected data (corrected_x): ", corrected_x)


# Some helper functions for reading bits from a file
def read_bits(filename: str, num_bits: int):
    """
        Reads a binary file and yields chunks of bits of a specified length.
        
        Parameters:
            filename (str): The path to the binary file to read.
            num_bits (int): The number of bits to yield in each chunk.
        Yields:
            numpy.ndarray: An array of bits of length `num_bits`.
        Example:
            for bits in read_bits("example.bin", 4):
                print(bits)
        Note:
            If the number of bits in the file is not a multiple of `num_bits`, the last chunk (incomplete chunk) would be ignored.
    """
    
    
    with open(filename, "rb") as f:
        file_in_bytes = f.read()
        
        # extract the 4-bit arrays from the line
        bits_array = np.unpackbits(np.frombuffer(file_in_bytes, dtype=np.uint8))
    
    bit_offset = 0
    bit_length = bits_array.shape[0]
    
    while bit_offset + num_bits <= bit_length:
        yield bits_array[bit_offset:bit_offset + num_bits]
        bit_offset += num_bits

# The function used to fix a corrupt file
def fix_corrupt_file(source_file: str, dest_file: str):
    """
        Fixes a corrupt file using Hamming (7,4) error correction code and writes the corrected data to a new file.
        
        Parameters:
            source_file (str): The path to the source file containing the corrupt data.
            dest_file (str): The path to the destination file where the corrected data will be written.
        Notes:
            - This function reads 16-bit chunks from the source file.
            - Each 16-bit chunk is split into two 7-bit arrays, ignoring the 7th and 15th bits.
            - The 7-bit arrays are corrected using the Hamming (7,4) error correction code.
            - The corrected 4-bit arrays are concatenated and written to the destination file as bytes.
    """
    
    with open(dest_file, "wb") as fw:
        for bits in read_bits(source_file, 16):
            # [TODO] Check the 2 7-bit array together to get 2 final 4-bit arrays
            # ignore the 7-th bits and 15-th bits
            # Hint: use the `fix_hamming_7_4` function
            first_7_bits = bits[0:7]
            second_7_bits = bits[8:15]

            fixed_first_4_bits = fix_hamming_7_4(first_7_bits)
            fixed_last_4_bits = fix_hamming_7_4(second_7_bits)
            
            # Concat them together
            fixed_8_bits = np.concatenate([fixed_first_4_bits, fixed_last_4_bits])
            
            # Write 2 fixed 4-bit arrays to the file (which is 1 byte)
            orig_content = convert_bit_array_to_btyes(fixed_8_bits.astype(np.byte))
            fw.write(orig_content)

# [TODO] Write down the name of the corrupt file and the destination file
corrupt_file_name = "C:/Users/tanth/OneDrive/Desktop/assign/corrupt_110000270.pdf"
destin_file_name = "ans.pdf"


fix_corrupt_file(corrupt_file_name, destin_file_name)