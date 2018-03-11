using System;
using System.Diagnostics;
using System.Linq;
using Alea;
using Alea.CSharp;

namespace DotnetosGPU.Alea
{
    class Program
    {
        static void Main(string[] args)
        {
            Action action;
            if (args.Length != 1)
            {
                Console.WriteLine("Missing command");
                return;
            }

            switch (args[0])
            {
                case "gpu":
                    action = RunGpu;
                    break;
                case "gpua":
                    action = RunGpuWithAutomaticMemoryManagement;
                    break;
                default:
                    Console.WriteLine($"Missing command. Expecting gpu | cpu");
                    return;
            }

            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine($"Warm up {i}/10");
                action?.Invoke();
            }

            GC.Collect(3);
            GC.WaitForPendingFinalizers();
            GC.Collect(3);

            var watch = Stopwatch.StartNew();
            action();
            watch.Stop();
            Console.WriteLine($"Done in {watch.Elapsed}");
        }

        public static void RunGpu()
        {
            var n = GetData(out var x, out var y);
            var result = new float[n];
            var gpu = Gpu.Default;
            var xDevice = gpu.Allocate<float>(n);
            var yDevice = gpu.Allocate<float>(n);
            var resultDevice = gpu.Allocate<float>(n);

            Gpu.Copy(x, xDevice);
            Gpu.Copy(y, yDevice);

            var lp = new LaunchParam(16, 256);
            gpu.Launch(Kernel, lp, resultDevice, xDevice, yDevice);
            Gpu.Copy(resultDevice, result);

            Gpu.Free(xDevice);
            Gpu.Free(yDevice);
            Gpu.Free(resultDevice);
        }


        [GpuManaged]
        public static void RunGpuWithAutomaticMemoryManagement()
        {
            var n = GetData(out var x, out var y);
            var result = new float[n];

            var gpu = Gpu.Default;
            var lp = new LaunchParam(16, 256);
            gpu.Launch(Kernel, lp, result, x, y);
        }

        private static int GetData(out float[] x, out float[] y)
        {
            int n = 10_000_000;
            x = Enumerable.Range(0, n).Select(i => i + 0.3f).ToArray();
            y = Enumerable.Range(0, n).Select(i => i + 0.6f).ToArray();

            return n;
        }

        private static void Kernel(float[] result, float[] x, float[] y)
        {
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;
            for (var i = start; i < x.Length; i += stride)
            {
                result[i] = x[i] + y[i];
            }
        }
    }
}
