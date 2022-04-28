#import "TorchModule.h"
#import <Libtorch-Lite/Libtorch-Lite.h>

@implementation TorchModule {
@protected
    torch::jit::mobile::Module _impl;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
    self = [super init];
    if (self) {
        try {
            _impl = torch::jit::_load_for_mobile(filePath.UTF8String);
//            _impl.eval();
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}

- (NSArray<NSNumber*>*)predictImage:(void*)imageBuffer {
    try {
        at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, 224, 224}, at::kFloat);
        
//        Tensor cpu() const;
//        Tensor cuda() const;
//        Tensor hip() const;
//        Tensor vulkan() const;
//        Tensor metal() const;

        printf("%d, %d, %d, %d, %d", tensor.is_cpu(), tensor.is_cuda(), tensor.is_hip(), tensor.is_vulkan(), tensor.is_metal());
        
        c10::InferenceMode mode;
        auto outputTensor = _impl.forward({tensor}).toTensor().cpu();
        float* floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) {
            return nil;
        }
        NSMutableArray* results = [[NSMutableArray alloc] init];
        for (int i = 0; i < 1000; i++) {
            [results addObject:@(floatBuffer[i])];
        }
        return [results copy];
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

@end


//#import "TorchModule.h"
////#import <Libtorch-Lite/Libtorch-Lite.h>
//// If it's built from source with xcode, comment out the line above
//// and use following headers
//#import <LibTorch/LibTorch.h>
// #include <torch/csrc/jit/mobile/import.h>
// #include <torch/csrc/jit/mobile/module.h>
// #include <torch/script.h>
//
//@implementation TorchModule {
// @protected
////  torch::jit::mobile::Module _impl;
//     torch::jit::script::Module _impl;
//}
//
//// 注意：Libtorch-Lite
//- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
//  self = [super init];
//  if (self) {
//    try {
////        _impl = torch::jit::_load_for_mobile(filePath.UTF8String);
//        _impl = torch::jit::load(filePath.UTF8String);
//        _impl.eval();
//    } catch (const std::exception& exception) {
//      NSLog(@"%s", exception.what());
//      return nil;
//    }
//  }
//  return self;
//}
//
//- (NSArray<NSNumber*>*)predictImage:(void*)imageBuffer {
//    try {
//        at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, 224, 224}, at::kFloat);
////        c10::InferenceMode guard;
//        auto outputTensor = _impl.forward({tensor}).toTensor();
//        float* floatBuffer = outputTensor.data_ptr<float>();
//        if (!floatBuffer) {
//            return nil;
//        }
//        NSMutableArray* results = [[NSMutableArray alloc] init];
//        for (int i = 0; i < 1000; i++) {
//            [results addObject:@(floatBuffer[i])];
//        }
//        return [results copy];
//    } catch (const std::exception& exception) {
//        NSLog(@"%s", exception.what());
//    }
//    return nil;
//}
//
//@end
//
