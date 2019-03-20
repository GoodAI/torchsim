// Temporary workaround to load untyped JS libraries (https://stackoverflow.com/a/50516783/2203164)
// declare module '*';

interface Collection<T> { }

interface List<T> extends Collection<T> {
    [index: number]: T;
    length: number;
}

interface Dictionary<T> extends Collection<T> {
    [index: string]: T;
}

declare module 'worker-loader!*' {
    class WebpackWorker extends Worker {
        constructor();
    }

    export = WebpackWorker;
}