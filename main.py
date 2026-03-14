import regex as re

class MiniTokenizer:
    def __init__(self):
        # GPT-2 Regex pattern: tách chữ, số, dấu câu và khoảng trắng
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.merges = {} # (int, int) -> int
        self.vocab = {i: bytes([i]) for i in range(256)} # int -> bytes
        self.special_tokens = {"<|endoftext|>": 1000}

    def _get_stats(self, ids, counts):
        """Thống kê tần suất các cặp ID xuất hiện cạnh nhau"""
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1

    def _merge(self, ids, pair, idx):
        """Thay thế cặp (p1, p2) bằng ID mới 'idx' trong danh sách ids"""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, vocab_size):
        num_merges = vocab_size - 256
        # Bước 1: Tiền xử lý bằng Regex
        text_chunks = re.findall(self.pat, text)
        ids_chunks = [list(ch.encode("utf-8")) for ch in text_chunks]

        # Bước 2: Vòng lặp hợp nhất (Training)
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids_chunks:
                self._get_stats(chunk_ids, stats)
            
            if not stats: break
            
            # Tìm cặp xuất hiện nhiều nhất
            pair = max(stats, key=stats.get)
            idx = 256 + i
            
            # Cập nhật các chunk và lưu luật merge
            ids_chunks = [self._merge(chunk_ids, pair, idx) for chunk_ids in ids_chunks]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            print(f"Hợp nhất {pair} thành ID {idx}")

    def encode(self, text):
        """Biến văn bản thành danh sách ID"""
        text_chunks = re.findall(self.pat, text)
        all_ids = []
        
        for chunk in text_chunks:
            chunk_ids = list(chunk.encode("utf-8"))
            # Áp dụng các luật merge đã học theo đúng thứ tự
            for pair, idx in self.merges.items():
                chunk_ids = self._merge(chunk_ids, pair, idx)
            all_ids.extend(chunk_ids)
            
        return all_ids

    def decode(self, ids):
        """Biến danh sách ID ngược lại thành văn bản"""
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.special_tokens.values():
                # Xử lý special token (tạm thời bỏ qua hoặc in ra text)
                for s, s_id in self.special_tokens.items():
                    if s_id == idx: part_bytes.append(s.encode("utf-8"))
        
        text_bytes = b"".join(part_bytes)
        return text_bytes.decode("utf-8", errors="replace")

# --- CHẠY THỬ ---
raw_text = "I don't blame programmers for still finding the whole thing mysterious, even 30 years after Unicode's inception."
tokenizer = MiniTokenizer()

print("--- Đang huấn luyện ---")
tokenizer.train(raw_text, vocab_size=270) # Thử 14 lần merge

print("\n--- Chạy thử Nghiệm ---")
test_sentence = "programmers finding Unicode!"
encoded = tokenizer.encode(test_sentence)
decoded = tokenizer.decode(encoded)

print(f"Văn bản gốc: {test_sentence}")
print(f"Token IDs: {encoded}")
print(f"Giải mã: {decoded}")