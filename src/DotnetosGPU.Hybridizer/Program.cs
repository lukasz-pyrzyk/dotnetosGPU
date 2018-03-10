using Hybridizer.Runtime.CUDAImports;
using System.Threading.Tasks;

namespace DotnetosGPU.Hybridizer
{
    public class Program
    {
        [EntryPoint("run")]
        public static void Run(int n, double[] a, double[] b, double[] results)
        {
            Parallel.For(0, n, i =>
            {
                results[i] = a[i] + b[i];
            });
        }

        public static void Main()
        {
            const int n = 1_000_000;
            var a = new double[n];
            var b = new double[n];
            var results = new double[n];

            cuda.GetDeviceProperties(out var prop, 0);
            HybRunner runner = HybRunner.Cuda("HelloWorld_CUDA.dll")
                .SetDistrib(prop.multiProcessorCount * 16, 128);

            // create a wrapper object to call GPU methods instead of C#
            dynamic wrapped = runner.Wrap(new Program());

            // run the method on GPU
            wrapped.Run(n, a, b, results);
        }
    }
}
