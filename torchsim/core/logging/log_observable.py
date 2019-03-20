import traceback

from torchsim.gui.observer_system import TextObservable


class LogObservable(TextObservable):
    text: str = ""

    def get_data(self):
        return self.text

    def log(self, line: str):
        self.text += line + '<br/>'  # os.linesep

    def log_last_exception(self):
        message = traceback.format_exc()
        html = f"""
            <div class="alert alert-danger">
                    <h4>Exception thrown:</h4>
                <pre>{message}</pre>
            </div>
        """
        self.log(html)
