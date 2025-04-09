#ifndef HLI_TYPE_INFER_OP_INTERFACES
#define HLI_TYPE_INFER_OP_INTERFACES

include "mlir/IR/OpBase.td"

def TypeInferInterface : OpInterface<"TypeInferInterface"> {
  let description = [{
    This is the description of the interface. It provides concrete information
    on the semantics of the interface, and how it may be used by the compiler.
  }];

  let methods = [
    InterfaceMethod<[{
      This method represents a simple non-static interface method with no
      inputs, and a void return type. This method is required to be implemented
      by all operations implementing this interface. This method roughly
      correlates to the following on an operation implementing this interface:

      ```c++
      class ConcreteOp ... {
      public:
        void nonStaticMethod();
      };
      ```
    }], "void", "nonStaticMethod"
    >,

    // InterfaceMethod<[{
    //   This method represents a non-static interface method with a non-void
    //   return value, as well as an `unsigned` input named `i`. This method is
    //   required to be implemented by all operations implementing this interface.
    //   This method roughly correlates to the following on an operation
    //   implementing this interface:

    //   ```c++
    //   class ConcreteOp ... {
    //   public:
    //     Value nonStaticMethodWithParams(unsigned i);
    //   };
    //   ```
    // }], "mlir::Value", "nonStaticMethodWithParams", (ins "unsigned":$i)
    // >,

    StaticInterfaceMethod<[{
      This method represents a static interface method with no inputs, and a
      void return type. This method is required to be implemented by all
      operations implementing this interface. This method roughly correlates
      to the following on an operation implementing this interface:

      ```c++
      class ConcreteOp ... {
      public:
        static void staticMethod();
      };
      ```
    }], "void", "staticMethod"
    >,

    // StaticInterfaceMethod<[{
    //   This method corresponds to a static interface method that has an explicit
    //   implementation of the method body. Given that the method body has been
    //   explicitly implemented, this method should not be defined by the operation
    //   implementing this method. This method merely takes advantage of properties
    //   already available on the operation, in this case its `build` methods. This
    //   method roughly correlates to the following on the interface `Model` class:

    //   ```c++
    //   struct InterfaceTraits {
    //     /// ... The `Concept` class is elided here ...

    //     template <typename ConcreteOp>
    //     struct Model : public Concept {
    //       Operation *create(OpBuilder &builder, Location loc) const override {
    //         return builder.create<ConcreteOp>(loc);
    //       }
    //     }
    //   };
    //   ```

    //   Note above how no modification is required for operations implementing an
    //   interface with this method.
    // }],
    //   "mlir::Operation *", "staticCreateWithBody", (ins "mlir::OpBuilder &":$builder, "mlir::Location":$loc),
    //   /*methodBody=*/[{
    //     return builder.create<ConcreteOp>(loc);
    // }]>,

    // InterfaceMethod<[{
    //   This method represents a non-static method that has an explicit
    //   implementation of the method body. Given that the method body has been
    //   explicitly implemented, this method should not be defined by the operation
    //   implementing this method. This method merely takes advantage of properties
    //   already available on the operation, in this case its `build` methods. This
    //   method roughly correlates to the following on the interface `Model` class:

    //   ```c++
    //   struct InterfaceTraits {
    //     /// ... The `Concept` class is elided here ...

    //     template <typename ConcreteOp>
    //     struct Model : public Concept {
    //       unsigned getNumInputsAndOutputs(Operation *opaqueOp) const override {
    //         ConcreteOp op = cast<ConcreteOp>(opaqueOp);
    //         return op.getNumInputs() + op.getNumOutputs();
    //       }
    //     }
    //   };
    //   ```

    //   Note above how no modification is required for operations implementing an
    //   interface with this method.
    // }],
    //   "unsigned", "getNumInputsAndOutputsWithBody", (ins), /*methodBody=*/[{
    //     return $_op.getNumInputs() + $_op.getNumOutputs();
    // }]>,

    // InterfaceMethod<[{
    //   This method represents a non-static method that has a default
    //   implementation of the method body. This means that the implementation
    //   defined here will be placed in the trait class that is attached to every
    //   operation that implements this interface. This has no effect on the
    //   generated `Concept` and `Model` class. This method roughly correlates to
    //   the following on the interface `Trait` class:

    //   ```c++
    //   template <typename ConcreteOp>
    //   class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
    //   public:
    //     bool isSafeToTransform() {
    //       ConcreteOp op = cast<ConcreteOp>(this->getOperation());
    //       return op.getProperties().hasFlag;
    //     }
    //   };
    //   ```

    //   As detailed in [Traits](Traits.md), given that each operation implementing
    //   this interface will also add the interface trait, the methods on this
    //   interface are inherited by the derived operation. This allows for
    //   injecting a default implementation of this method into each operation that
    //   implements this interface, without changing the interface class itself. If
    //   an operation wants to override this default implementation, it merely
    //   needs to implement the method and the derived implementation will be
    //   picked up transparently by the interface class.

    //   ```c++
    //   class ConcreteOp ... {
    //   public:
    //     bool isSafeToTransform() {
    //       // Here we can override the default implementation of the hook
    //       // provided by the trait.
    //     }
    //   };
    //   ```
    // }],
    //   "bool", "isSafeToTransformWithDefault", (ins), /*methodBody=*/[{}],
    //   /*defaultImplementation=*/[{
    //     return $_op.getProperties().hasFlag;
    // }]>,
  ];
}

#endif // HLI_TYPE_INFER_OP_INTERFACES