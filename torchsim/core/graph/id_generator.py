class IdGenerator:
    _last_node_id: int = 0

    def next_node_id(self):
        self._last_node_id += 1
        return self._last_node_id
